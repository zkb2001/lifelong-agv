"""
Generate solvable stress scenarios (position/task CSVs + JSON manifest).

Does not modify main copy.py. Extra obstacles are applied at runtime via
ENV.get_static_obstacles monkey-patch (stored in scenario JSON).
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCENARIO_DIR = Path(__file__).resolve().parent

START_POINTS = [
    ("Tiger", 1, 6),
    ("Dragon", 1, 10),
    ("Horse", 1, 14),
    ("Rabbit", 20, 6),
    ("Ox", 20, 10),
    ("Monkey", 20, 14),
]

END_POINTS = [
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

# Default 12 AGVs (same layout as agv_position.csv)
BASE_AGVS = [
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

# Extra free cells for fleets beyond 12 (non-overlapping with stations/AGVs).
# Layout supports up to 50 AGVs on the 21x21 warehouse grid.
EXTRA_AGV_SLOTS = [
    ("Prime", 4, 10, 0),
    ("Prowl", 8, 10, 0),
    ("Mirage", 13, 10, 180),
    ("Sunstreaker", 17, 10, 180),
    # 17–32
    ("HotRod", 2, 8, 0),
    ("Kup", 2, 12, 0),
    ("Arcee", 5, 3, 90),
    ("Chromia", 8, 3, 90),
    ("Elita", 11, 3, 90),
    ("Flamewar", 14, 3, 90),
    ("Moonracer", 17, 3, 90),
    ("Novastar", 19, 8, 180),
    ("Greenlight", 19, 12, 180),
    ("Strongarm", 5, 18, 270),
    ("Windblade", 8, 18, 270),
    ("Override", 11, 18, 270),
    ("Thunderblast", 14, 18, 270),
    ("Slipstream", 17, 18, 270),
    ("Nautica", 4, 5, 90),
    ("Velocity", 16, 5, 90),
    # 33–50
    ("Roadbuster", 2, 4, 0),
    ("Whirl", 2, 16, 0),
    ("Topspin", 19, 4, 180),
    ("TwinTwist", 19, 16, 180),
    ("Broadside", 7, 5, 90),
    ("Sandstorm", 10, 5, 90),
    ("Springer", 7, 15, 270),
    ("UltraMagnus", 10, 15, 270),
    ("Metroplex", 13, 5, 90),
    ("Fortress", 13, 15, 270),
    ("Omega", 4, 15, 270),
    ("AlphaTrion", 16, 15, 270),
    ("Perceptor", 5, 10, 0),
    ("Blaster", 15, 10, 180),
    ("Soundwave", 7, 8, 0),
    ("Shockwave", 14, 12, 180),
    ("Starscream", 3, 8, 0),
    ("Skywarp", 18, 12, 180),
]

LEFT_STATIONS = ["Tiger", "Dragon", "Horse"]
RIGHT_STATIONS = ["Rabbit", "Ox", "Monkey"]
LEFTISH_DESTS = ["Beijing", "Shanghai", "Suzhou", "Hangzhou", "Nanjing", "Wuhan", "Changsha", "Guangzhou"]
RIGHTISH_DESTS = ["Chengdu", "Xiamen", "Kunming", "Urumqi", "Shenzhen", "Dalian", "Tianjin", "Chongqing"]
ALL_STATIONS = [s[0] for s in START_POINTS]
ALL_DESTS = [e[0] for e in END_POINTS]


def _reserved_cells() -> set:
    cells = set()
    for _, x, y in START_POINTS + END_POINTS:
        cells.add((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cells.add((x + dx, y + dy))
    for _, x, y, _ in BASE_AGVS + EXTRA_AGV_SLOTS:
        cells.add((x, y))
    return cells


def select_agvs(n_agvs: int) -> List[Tuple[str, int, int, int]]:
    """Pick n AGVs without overlapping poses (supports up to 50)."""
    max_fleet = len(BASE_AGVS) + len(EXTRA_AGV_SLOTS)
    if n_agvs < 1 or n_agvs > max_fleet:
        raise ValueError(f"n_agvs must be in 1..{max_fleet}, got {n_agvs}")
    if n_agvs <= 12:
        # Prefer alternating top/bottom for balance
        order = [
            BASE_AGVS[0],
            BASE_AGVS[6],
            BASE_AGVS[1],
            BASE_AGVS[7],
            BASE_AGVS[2],
            BASE_AGVS[8],
            BASE_AGVS[3],
            BASE_AGVS[9],
            BASE_AGVS[4],
            BASE_AGVS[10],
            BASE_AGVS[5],
            BASE_AGVS[11],
        ]
        return order[:n_agvs]
    return list(BASE_AGVS) + list(EXTRA_AGV_SLOTS[: max(0, n_agvs - 12)])


def build_corridor_obstacles(seed: int, density: str = "mild") -> List[Tuple[int, int]]:
    """
    Vertical bottleneck near x=10/11 with 1–2 gaps — solvable but constrains crossing.
    density: mild | hard
    """
    rng = random.Random(seed)
    reserved = _reserved_cells()
    obs: List[Tuple[int, int]] = []
    gap_ys = {10, 11} if density == "mild" else {10}
    for x in (10, 11) if density == "hard" else (10,):
        for y in range(2, 20):
            if y in gap_ys:
                continue
            cell = (x, y)
            if cell in reserved:
                continue
            # mild: random thin wall; hard: denser
            if density == "hard" or rng.random() < 0.85:
                obs.append(cell)
    # Sprinkle a few aisle blockers away from stations
    n_extra = 6 if density == "mild" else 12
    candidates = [
        (x, y)
        for x in range(2, 20)
        for y in range(2, 20)
        if (x, y) not in reserved and (x, y) not in obs and x not in (1, 20)
    ]
    rng.shuffle(candidates)
    for c in candidates[:n_extra]:
        obs.append(c)
    return obs


def write_position_csv(path: Path, n_agvs: int) -> None:
    agvs = select_agvs(n_agvs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "name", "x", "y", "pitch"])
        for name, x, y in START_POINTS:
            w.writerow(["start_point", name, x, y, ""])
        for name, x, y in END_POINTS:
            w.writerow(["end_point", name, x, y, ""])
        for name, x, y, pitch in agvs:
            w.writerow(["agv", name, x, y, pitch])


def _write_tasks(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["task_id", "start_point", "end_point", "priority", "remaining_time"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _tid(station: str, counters: Dict[str, int]) -> str:
    counters[station] = counters.get(station, 0) + 1
    return f"{station}-{counters[station]}"


def gen_tasks(
    pattern: str,
    n_tasks: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    counters: Dict[str, int] = {}
    rows: List[dict] = []

    def add(station: str, dest: str, priority: str = "Normal", rt=None):
        rows.append(
            {
                "task_id": _tid(station, counters),
                "start_point": station,
                "end_point": dest,
                "priority": priority,
                "remaining_time": rt if rt is not None else "None",
            }
        )

    if pattern == "uniform":
        for _ in range(n_tasks):
            add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS))

    elif pattern == "clustered_dest":
        hub = rng.choice(["Xiamen", "Wuhan", "Chengdu"])
        secondary = rng.choice([d for d in ALL_DESTS if d != hub])
        for i in range(n_tasks):
            # 55% hub keeps congestion signal without extreme A* blow-up
            dest = hub if i < int(0.55 * n_tasks) else secondary
            add(rng.choice(ALL_STATIONS), dest)

    elif pattern == "clustered_src":
        hot = rng.sample(ALL_STATIONS, 2)
        for i in range(n_tasks):
            st = hot[i % 2] if i < int(0.8 * n_tasks) else rng.choice(ALL_STATIONS)
            add(st, rng.choice(ALL_DESTS))

    elif pattern == "urgent_burst":
        # Front-load several Urgent with tight remaining_time
        n_u = max(4, n_tasks // 5)
        for i in range(n_tasks):
            if i < n_u:
                add(
                    rng.choice(ALL_STATIONS),
                    rng.choice(ALL_DESTS),
                    "Urgent",
                    80 + 25 * i,
                )
            else:
                add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS))

    elif pattern == "fifo_pressure":
        # Deep queues per station; urgent buried mid-queue (FIFO head blocks)
        per = max(4, n_tasks // len(ALL_STATIONS))
        for st in ALL_STATIONS:
            for k in range(per):
                if k == per // 2:
                    add(st, rng.choice(ALL_DESTS), "Urgent", 120 + 10 * k)
                else:
                    add(st, rng.choice(ALL_DESTS))
        rows = rows[:n_tasks]

    elif pattern == "cross_flow":
        for i in range(n_tasks):
            if i % 2 == 0:
                add(rng.choice(LEFT_STATIONS), rng.choice(RIGHTISH_DESTS))
            else:
                add(rng.choice(RIGHT_STATIONS), rng.choice(LEFTISH_DESTS))

    elif pattern == "far_cross":
        far_left_dest = ["Shenzhen", "Dalian", "Tianjin", "Chongqing"]
        far_right_dest = ["Beijing", "Shanghai", "Hangzhou", "Suzhou"]
        for i in range(n_tasks):
            if i % 2 == 0:
                add(rng.choice(LEFT_STATIONS), rng.choice(far_left_dest))
            else:
                add(rng.choice(RIGHT_STATIONS), rng.choice(far_right_dest))

    elif pattern == "two_hubs":
        hubs = ["Xiamen", "Beijing"]
        for i in range(n_tasks):
            add(rng.choice(ALL_STATIONS), hubs[i % 2])

    elif pattern == "mixed":
        # Blend: ~40% uniform, 30% cross, 20% cluster, 10% urgent
        for i in range(n_tasks):
            r = i / max(1, n_tasks)
            if r < 0.1:
                add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS), "Urgent", 100 + 30 * i)
            elif r < 0.4:
                add(rng.choice(LEFT_STATIONS), rng.choice(RIGHTISH_DESTS))
            elif r < 0.7:
                add(rng.choice(ALL_STATIONS), "Xiamen")
            else:
                add(rng.choice(ALL_STATIONS), rng.choice(ALL_DESTS))

    elif pattern == "sparse_longhaul":
        far_from_left = ["Shenzhen", "Dalian", "Tianjin", "Chongqing"]
        far_from_right = ["Beijing", "Shanghai", "Hangzhou", "Suzhou"]
        for _ in range(n_tasks):
            st = rng.choice(LEFT_STATIONS + RIGHT_STATIONS)
            dest = rng.choice(far_from_left if st in LEFT_STATIONS else far_from_right)
            add(st, dest)

    else:
        raise ValueError(f"Unknown pattern {pattern}")

    return rows[:n_tasks]


# ---- Scenario catalog (≥20 solvable-leaning designs) ----

SCENARIO_SPECS: List[Dict[str, Any]] = [
    # Controls
    dict(id="S01_ctrl_n12_uniform", intent="正常对照：12车均匀任务", n_agvs=12, obstacle="none", pattern="uniform", n_tasks=30, seed=101),
    dict(id="S02_ctrl_n12_mixed", intent="正常混合对照", n_agvs=12, obstacle="none", pattern="mixed", n_tasks=32, seed=102),
    # Fleet size
    dict(id="S03_agv4_uniform", intent="少车运力不足压力", n_agvs=4, obstacle="none", pattern="uniform", n_tasks=24, seed=201),
    dict(id="S04_agv8_uniform", intent="中等车队均匀", n_agvs=8, obstacle="none", pattern="uniform", n_tasks=28, seed=202),
    dict(id="S05_agv16_uniform", intent="多车：soft_cap/互扰", n_agvs=16, obstacle="none", pattern="uniform", n_tasks=36, seed=203),
    # Destination / source clustering
    dict(id="S06_cluster_dest", intent="同卸货点聚集→端点拥堵", n_agvs=12, obstacle="none", pattern="clustered_dest", n_tasks=28, seed=301),
    dict(id="S07_cluster_dest_agv4", intent="少车+聚集卸货（贪心易堵）", n_agvs=4, obstacle="none", pattern="clustered_dest", n_tasks=24, seed=302),
    dict(id="S08_cluster_dest_agv16", intent="多车涌向同卸货点（控规模可解）", n_agvs=16, obstacle="none", pattern="clustered_dest", n_tasks=24, seed=303),
    dict(id="S09_cluster_src", intent="同取货站突发FIFO", n_agvs=12, obstacle="none", pattern="clustered_src", n_tasks=30, seed=304),
    dict(id="S10_two_hubs", intent="双枢纽卸货竞争", n_agvs=12, obstacle="none", pattern="two_hubs", n_tasks=32, seed=305),
    # Urgency / FIFO
    dict(id="S11_urgent_burst", intent="紧急件密集（权重弱）", n_agvs=12, obstacle="none", pattern="urgent_burst", n_tasks=30, seed=401),
    dict(id="S12_fifo_pressure", intent="FIFO深队列埋紧急件", n_agvs=12, obstacle="none", pattern="fifo_pressure", n_tasks=36, seed=402),
    dict(id="S13_urgent_agv8", intent="中车队+紧急突发", n_agvs=8, obstacle="none", pattern="urgent_burst", n_tasks=28, seed=403),
    # Cross / longhaul
    dict(id="S14_cross_flow", intent="远距离左右交叉流", n_agvs=12, obstacle="none", pattern="cross_flow", n_tasks=32, seed=501),
    dict(id="S15_far_cross", intent="对角超长交叉", n_agvs=12, obstacle="none", pattern="far_cross", n_tasks=30, seed=502),
    dict(id="S16_sparse_longhaul", intent="稀疏长途任务", n_agvs=8, obstacle="none", pattern="sparse_longhaul", n_tasks=24, seed=503),
    dict(id="S17_cross_agv4", intent="少车交叉流易超时", n_agvs=4, obstacle="none", pattern="cross_flow", n_tasks=24, seed=504),
    # Narrow corridor
    dict(id="S18_narrow_uniform", intent="走廊瓶颈+均匀", n_agvs=12, obstacle="mild", pattern="uniform", n_tasks=28, seed=601),
    dict(id="S19_narrow_cross", intent="走廊瓶颈+交叉流", n_agvs=12, obstacle="mild", pattern="cross_flow", n_tasks=28, seed=602),
    dict(id="S20_narrow_hard_cluster", intent="窄通道+聚集卸货", n_agvs=8, obstacle="mild", pattern="clustered_dest", n_tasks=24, seed=603),
    dict(id="S21_narrow_urgent", intent="瓶颈+紧急件", n_agvs=12, obstacle="mild", pattern="urgent_burst", n_tasks=24, seed=604),
    dict(id="S22_narrow_agv16", intent="多车挤窄通道", n_agvs=12, obstacle="mild", pattern="cross_flow", n_tasks=28, seed=605),
    # Combined stress (still solvable-leaning)
    dict(id="S23_agv4_fifo_urgent", intent="少车+FIFO紧急压力", n_agvs=4, obstacle="none", pattern="fifo_pressure", n_tasks=24, seed=701),
    dict(id="S24_mixed_narrow", intent="混合任务+轻度瓶颈", n_agvs=12, obstacle="mild", pattern="mixed", n_tasks=30, seed=702),
    # Odd fleet sizes (with / without corridor obstacles)
    dict(id="S25_agv3_uniform", intent="奇数车队3无障碍", n_agvs=3, obstacle="none", pattern="uniform", n_tasks=24, seed=801),
    dict(id="S26_agv3_narrow", intent="奇数车队3有障碍", n_agvs=3, obstacle="mild", pattern="uniform", n_tasks=24, seed=802),
    dict(id="S27_agv5_uniform", intent="奇数车队5无障碍", n_agvs=5, obstacle="none", pattern="uniform", n_tasks=28, seed=803),
    dict(id="S28_agv5_narrow", intent="奇数车队5有障碍", n_agvs=5, obstacle="mild", pattern="cross_flow", n_tasks=28, seed=804),
    dict(id="S29_agv7_uniform", intent="奇数车队7无障碍", n_agvs=7, obstacle="none", pattern="mixed", n_tasks=28, seed=805),
    dict(id="S30_agv7_narrow", intent="奇数车队7有障碍", n_agvs=7, obstacle="mild", pattern="mixed", n_tasks=28, seed=806),
    dict(id="S31_agv9_uniform", intent="奇数车队9无障碍", n_agvs=9, obstacle="none", pattern="cross_flow", n_tasks=30, seed=807),
    dict(id="S32_agv9_narrow", intent="奇数车队9有障碍", n_agvs=9, obstacle="mild", pattern="cross_flow", n_tasks=30, seed=808),
    dict(id="S33_agv11_uniform", intent="奇数车队11无障碍", n_agvs=11, obstacle="none", pattern="urgent_burst", n_tasks=30, seed=809),
    dict(id="S34_agv11_narrow", intent="奇数车队11有障碍", n_agvs=11, obstacle="mild", pattern="urgent_burst", n_tasks=30, seed=810),
    dict(id="S35_agv13_uniform", intent="奇数车队13无障碍", n_agvs=13, obstacle="none", pattern="clustered_dest", n_tasks=32, seed=811),
    dict(id="S36_agv13_narrow", intent="奇数车队13有障碍", n_agvs=13, obstacle="mild", pattern="clustered_dest", n_tasks=32, seed=812),
    dict(id="S37_agv15_uniform", intent="奇数车队15无障碍", n_agvs=15, obstacle="none", pattern="fifo_pressure", n_tasks=32, seed=813),
    dict(id="S38_agv15_narrow", intent="奇数车队15有障碍", n_agvs=15, obstacle="mild", pattern="fifo_pressure", n_tasks=32, seed=814),
    # Massive fleets (crowded)
    dict(id="S39_agv50_crowd", intent="50车无障碍拥挤", n_agvs=50, obstacle="none", pattern="cross_flow", n_tasks=40, seed=821),
    dict(id="S40_agv50_narrow_crowd", intent="50车有障碍拥挤", n_agvs=50, obstacle="mild", pattern="cross_flow", n_tasks=40, seed=822),
]

# Pattern fallback when base scenario uses custom hard_extra task CSVs
HARD_EXTRA_PATTERN_MAP = {
    "H01_tight_urgent": "urgent_burst",
    "H02_fifo_block_urgent": "fifo_pressure",
    "H03_heavy_hub": "clustered_dest",
    "H04_narrow_cross_urgent": "mixed",
    "H05_agv4_heavy_cross": "cross_flow",
    "H06_far_urgent_vs_hub": "far_cross",
}


def _materialize_spec(
    spec: Dict[str, Any],
    out_dir: Path,
    max_sim_time: int,
    wall_timeout: float,
) -> Dict[str, Any]:
    sid = spec["id"]
    pos_path = out_dir / f"{sid}_position.csv"
    task_path = out_dir / f"{sid}_task.csv"
    write_position_csv(pos_path, int(spec["n_agvs"]))

    pattern = spec["pattern"]
    rows = gen_tasks(pattern, int(spec["n_tasks"]), int(spec["seed"]))
    _write_tasks(task_path, rows)

    obstacles: List[List[int]] = []
    if spec["obstacle"] == "mild":
        obstacles = [list(p) for p in build_corridor_obstacles(int(spec["seed"]) + 17, "mild")]
    elif spec["obstacle"] == "hard":
        obstacles = [list(p) for p in build_corridor_obstacles(int(spec["seed"]) + 17, "hard")]

    meta = {
        "id": sid,
        "intent": spec["intent"],
        "n_agvs": spec["n_agvs"],
        "obstacle_mode": spec["obstacle"],
        "obstacle_seed": int(spec["seed"]) + 17,
        "task_pattern": pattern,
        "n_tasks": len(rows),
        "seed": spec["seed"],
        "position_csv": str(pos_path.name),
        "task_csv": str(task_path.name),
        "extra_obstacles": obstacles,
        "max_sim_time": max_sim_time,
        "wall_timeout": wall_timeout,
    }
    (out_dir / f"{sid}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def generate_all_scenarios(
    out_dir: Optional[Path] = None,
    max_sim_time: int = 900,
    wall_timeout: float = 90.0,
    preserve_hard: bool = True,
) -> List[Dict[str, Any]]:
    """Generate S-scenario catalog. By default keeps existing H* hard scenarios in manifest."""
    out_dir = Path(out_dir or SCENARIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    preserved: List[Dict[str, Any]] = []
    man_path = out_dir / "manifest.json"
    if preserve_hard and man_path.exists():
        old = json.loads(man_path.read_text(encoding="utf-8"))
        preserved = [c for c in old.get("scenarios", []) if str(c.get("id", "")).startswith("H")]

    catalog = [
        _materialize_spec(spec, out_dir, max_sim_time, wall_timeout) for spec in SCENARIO_SPECS
    ]
    # Append preserved H scenes (files already on disk)
    seen = {c["id"] for c in catalog}
    for h in preserved:
        if h["id"] not in seen:
            catalog.append(h)
            seen.add(h["id"])

    manifest = {
        "n_scenarios": len(catalog),
        "max_sim_time": max_sim_time,
        "wall_timeout": wall_timeout,
        "scenarios": catalog,
    }
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return catalog


def ensure_extended_scenarios(
    out_dir: Optional[Path] = None,
    max_sim_time: int = 900,
    wall_timeout: float = 90.0,
) -> List[Dict[str, Any]]:
    """Materialize S25–S40 only and merge into manifest without wiping H*."""
    out_dir = Path(out_dir or SCENARIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in man.get("scenarios", [])}
    else:
        man = {"max_sim_time": max_sim_time, "wall_timeout": wall_timeout, "scenarios": []}
        by_id = {}

    extended = [s for s in SCENARIO_SPECS if s["id"] >= "S25"]
    for spec in extended:
        meta = _materialize_spec(spec, out_dir, max_sim_time, wall_timeout)
        by_id[meta["id"]] = meta

    catalog = list(by_id.values())
    catalog.sort(key=lambda c: c["id"])
    man["scenarios"] = catalog
    man["n_scenarios"] = len(catalog)
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    return [by_id[s["id"]] for s in extended]


def load_scenario_manifest(out_dir: Optional[Path] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir or SCENARIO_DIR)
    path = out_dir / "manifest.json"
    if not path.exists():
        generate_all_scenarios(out_dir)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    cats = generate_all_scenarios()
    print(f"Generated {len(cats)} scenarios under {SCENARIO_DIR}")
    for c in cats:
        print(f"  {c['id']}: agvs={c['n_agvs']} tasks={c['n_tasks']} pattern={c['task_pattern']} obs={c['obstacle_mode']}")
