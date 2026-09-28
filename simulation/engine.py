
from typing import Any, Tuple, List, Optional
import json
import csv
import random
import pandas as pd
import heapq
import sys
import io
import time as _time
from collections import defaultdict
import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# Congestion control (holding / endpoint capacity / A* budgets)
# Toggle: set AGV_CONGESTION=0 to disable; or Simulation.congestion_control=False
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "off", "no")


CONGESTION_CONTROL = _env_flag("AGV_CONGESTION", True)
# Hard pipeline cap = n_slots + EXTRA (legacy soft_cap was len(agvs) → Extreme/H03 爆炸)
ENDPOINT_CAPACITY_EXTRA = int(os.environ.get("AGV_ENDPOINT_CAP_EXTRA", "1"))
ASTAR_MAX_PATH_LEN = int(os.environ.get("AGV_ASTAR_MAX_PATH", "80"))
ASTAR_MAX_VISITED = int(os.environ.get("AGV_ASTAR_MAX_VISITED", "2500"))
ASTAR_MAX_FRONTIER = int(os.environ.get("AGV_ASTAR_MAX_FRONTIER", "8000"))
ASTAR_WALL_BUDGET_S = float(os.environ.get("AGV_ASTAR_WALL_S", "0.35"))
# Rolling reservation: only treat other AGVs as moving obstacles within plan_t0 + N steps.
# 0 = legacy full-trajectory freeze at path end. Set via MOVING_OBSTACLE_HORIZON or patch_moving_obstacle_horizon().
MOVING_OBSTACLE_HORIZON = int(os.environ.get("MOVING_OBSTACLE_HORIZON", "0") or "0")
# Warehouse playable cells are 1..20 (matches display.py 20x20). Never allow 21 —
# that cell draws off-screen and looks like AGVs vanishing / re-entering from edges.
GRID_MIN = 1
GRID_MAX = 20
DEFAULT_GRID_SIZE = (GRID_MAX, GRID_MAX)


def _in_playable(x: int, y: int) -> bool:
    return GRID_MIN <= int(x) <= GRID_MAX and GRID_MIN <= int(y) <= GRID_MAX


def truncate_path_to_horizon(path, t0: int, horizon: int):
    """Keep only poses with timestamp <= t0+horizon. horizon<=0 → no truncate."""
    if not path:
        return []
    h = int(horizon)
    if h <= 0:
        return [tuple(p) for p in path]
    t_cap = int(t0) + h
    out = []
    for p in path:
        try:
            ts = int(p[2])
        except Exception:  # noqa: BLE001
            continue
        if ts <= t_cap:
            out.append(tuple(p))
    return out


def truncate_steps_to_horizon(steps, t0: int, horizon: int):
    """Keep step rows with timestamp <= t0+horizon. horizon<=0 → no truncate."""
    if not steps:
        return []
    h = int(horizon)
    if h <= 0:
        return [dict(s) for s in steps]
    t_cap = int(t0) + h
    out = []
    for s in steps:
        try:
            ts = int(s.get("timestamp", -1))
        except Exception:  # noqa: BLE001
            continue
        if ts <= t_cap:
            out.append(dict(s))
    return out


def _mo_horizon_active(plan_t0: int, ts: int, horizon: Optional[int] = None) -> bool:
    h = MOVING_OBSTACLE_HORIZON if horizon is None else int(horizon)
    if not h or h <= 0:
        return True
    return int(ts) <= int(plan_t0) + int(h)


def _traj_pose_at_timestamp(traj, t_query: int):
    """Return pose tuple at exact timestamp, or None if no planned pose at t.

    Prefers index-aligned ``traj[t]`` when ``traj[t][2]==t``; otherwise binary
    search by the timestamp field (paths can desync after kinematic expand).
    """
    if not traj:
        return None
    t_query = int(t_query)
    n = len(traj)
    if 0 <= t_query < n and len(traj[t_query]) >= 3 and int(traj[t_query][2]) == t_query:
        return traj[t_query]
    lo, hi = 0, n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        p = traj[mid]
        if len(p) < 3:
            lo = mid + 1
            continue
        pt = int(p[2])
        if pt == t_query:
            return p
        if pt < t_query:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def _traj_pose_floor(traj, t_query: int):
    """Latest pose with timestamp <= t_query (hold through gaps)."""
    exact = _traj_pose_at_timestamp(traj, t_query)
    if exact is not None:
        return exact
    if not traj:
        return None
    t_query = int(t_query)
    n = len(traj)
    lo, hi = 0, n - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        p = traj[mid]
        if len(p) < 3:
            lo = mid + 1
            continue
        pt = int(p[2])
        if pt <= t_query:
            best = p
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def mo_occupied_cell(
    traj,
    t_query: int,
    plan_t0: int = 0,
    horizon: Optional[int] = None,
):
    """Cell occupied by another AGV at ``t_query``, or None if free for planning.

    The AGV is a physical body:
    - Explicit planned pose at ``t_query`` always blocks.
    - Missing ticks inside the trajectory span still occupy the previous cell
      (gaps must not look free — that let followers drive through in-place turns).
    - After the path ends they remain on the last cell until a new path is
      committed. Rolling-horizon vanish was starvation sugar and caused CSV
      vertex collisions. ``plan_t0`` / ``horizon`` are kept for API compat.
    """
    del plan_t0, horizon
    if not traj:
        return None
    t_query = int(t_query)
    pose = _traj_pose_floor(traj, t_query)
    if pose is not None:
        return pose[:2]
    return None


def path_has_spacetime_conflict(new_path, moving_obstacles, self_name=None) -> bool:
    """True if ``new_path`` vertex- or edge-conflicts with any other trajectory.

    Also rejects parking on a cell that another AGV already reserved at any
    later timestamp: after commit we freeze on the last cell, so ending on a
    future reservation becomes a CSV collision once idle padding runs.
    """
    if not new_path:
        return True
    for name, trajectory in (moving_obstacles or {}).items():
        if self_name and name == self_name:
            continue
        if not trajectory:
            continue
        for state in new_path:
            other = mo_occupied_cell(trajectory, int(state[2]))
            if other is not None and tuple(other) == tuple(state[:2]):
                return True
        for i in range(1, len(new_path)):
            t0 = int(new_path[i - 1][2])
            t1 = int(new_path[i][2])
            if t1 != t0 + 1:
                continue
            our_a, our_b = tuple(new_path[i - 1][:2]), tuple(new_path[i][:2])
            if our_a == our_b:
                continue
            their_a = mo_occupied_cell(trajectory, t0)
            their_b = mo_occupied_cell(trajectory, t1)
            if their_a == our_b and their_b == our_a:
                return True
    # Terminal freeze vs others' near-future reserved poses. Full-horizon forever
    # over-rejects (long foreign paths reserve half the map); idle-on-claimed
    # repair in update_agv_state covers the long tail.
    freeze_window = 60
    last_xy = tuple(new_path[-1][:2])
    last_t = int(new_path[-1][2])
    t_lim = last_t + freeze_window
    for name, trajectory in (moving_obstacles or {}).items():
        if self_name and name == self_name:
            continue
        if not trajectory:
            continue
        for p in trajectory:
            if len(p) < 3:
                continue
            pt = int(p[2])
            if last_t <= pt <= t_lim and tuple(p[:2]) == last_xy:
                return True
    return False


# ---------------------------------------------------------------------------
# Neural A* (residual heuristic + optional conflict cost inflate)
# Residual default ON (compatible with congestion). Conflict pred default OFF.
# Toggle: AGV_RESIDUAL=0 / AGV_CONFLICT_PRED=1; or set_neural_astar(...)
# ---------------------------------------------------------------------------
RESIDUAL_HEURISTIC = _env_flag("AGV_RESIDUAL", True)
RESIDUAL_MODE = os.environ.get("AGV_RESIDUAL_MODE", "residual").strip().lower()
if RESIDUAL_MODE not in ("manhattan", "residual", "focal"):
    RESIDUAL_MODE = "residual"
CONFLICT_PRED = _env_flag("AGV_CONFLICT_PRED", False)
CONFLICT_INFLATE = float(os.environ.get("AGV_CONFLICT_INFLATE", "1.5"))
CONFLICT_THRESH = float(os.environ.get("AGV_CONFLICT_THRESH", "0.5"))
# Minimal online hook: only adjacent cells; hard cap inferences per A* search
_CONFLICT_LOCAL_R = int(os.environ.get("AGV_CONFLICT_RADIUS", "1"))
_CONFLICT_MAX_CALLS = int(os.environ.get("AGV_CONFLICT_MAX_CALLS", "48"))
_conflict_inflate_cache = {}
_conflict_occ_cache = {}
_conflict_search_calls = 0
_conflict_t0 = None
_residual_h = None
_conflict_p = None
_residual_load_attempted = False
_conflict_load_attempted = False


def set_neural_astar(residual=None, residual_mode=None, conflict=None):
    """Runtime toggle used by benchmarks (no process restart)."""
    global RESIDUAL_HEURISTIC, RESIDUAL_MODE, CONFLICT_PRED
    if residual is not None:
        RESIDUAL_HEURISTIC = bool(residual)
    if residual_mode is not None:
        m = str(residual_mode).strip().lower()
        if m in ("manhattan", "residual", "focal"):
            RESIDUAL_MODE = m
    if conflict is not None:
        CONFLICT_PRED = bool(conflict)


def _lazy_residual():
    global _residual_h, _residual_load_attempted
    if not RESIDUAL_HEURISTIC:
        return None
    if _residual_h is not None:
        return _residual_h
    if _residual_load_attempted:
        return None
    _residual_load_attempted = True
    try:
        from ml_research.integrate.residual_heuristic import get_residual_heuristic
        _residual_h = get_residual_heuristic()
        if _residual_h is not None:
            print(f"[A*] residual heuristic ON ckpt={_residual_h.ckpt.name} mode={RESIDUAL_MODE}", flush=True)
        return _residual_h
    except Exception as e:  # noqa: BLE001
        print(f"[A*] residual heuristic unavailable: {e}", flush=True)
        return None


def _lazy_conflict():
    global _conflict_p, _conflict_load_attempted
    if not CONFLICT_PRED:
        return None
    if _conflict_p is not None:
        return _conflict_p
    if _conflict_load_attempted:
        return None
    _conflict_load_attempted = True
    try:
        from ml_research.integrate.residual_heuristic import get_conflict_predictor
        _conflict_p = get_conflict_predictor(force=True)
        if _conflict_p is not None:
            print(f"[A*] conflict predictor ON ckpt={_conflict_p.ckpt.name}", flush=True)
        return _conflict_p
    except Exception as e:  # noqa: BLE001
        print(f"[A*] conflict predictor unavailable: {e}", flush=True)
        return None


def astar_heuristic(pos1, pos2) -> float:
    """A* f-score heuristic; residual when enabled, else Manhattan. Assignment costs stay pure Manhattan."""
    if not RESIDUAL_HEURISTIC:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
    rh = _lazy_residual()
    if rh is None:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))
    return float(rh.h(pos1, pos2, mode=RESIDUAL_MODE))


def _begin_astar_search(start_t=None):
    global _conflict_inflate_cache, _conflict_occ_cache, _conflict_search_calls, _conflict_t0
    rh = _lazy_residual() if RESIDUAL_HEURISTIC else None
    if rh is not None:
        rh.begin_search()
    _conflict_inflate_cache = {}
    _conflict_occ_cache = {}
    _conflict_search_calls = 0
    _conflict_t0 = int(start_t) if start_t is not None else None


def _conflict_occ_pair(timestamp, moving_obstacles, self_name, x, y):
    """Build/cached occupancy grids for conflict net around (x,y) at timestamp."""
    key = (timestamp, self_name)
    cached = _conflict_occ_cache.get(key)
    if cached is not None:
        return cached
    occ_now = np.zeros((22, 22), dtype=np.float32)
    occ_fut = np.zeros((22, 22), dtype=np.float32)
    for name, traj in moving_obstacles.items():
        if name == self_name or not traj:
            continue
        t1 = timestamp + 1
        p1 = traj[t1][:2] if t1 < len(traj) else traj[-1][:2]
        if _in_playable(p1[0], p1[1]):
            occ_now[p1[1], p1[0]] = 1.0
        for tt in range(timestamp + 1, timestamp + 4):
            p = traj[tt][:2] if tt < len(traj) else traj[-1][:2]
            if _in_playable(p[0], p[1]):
                occ_fut[p[1], p[0]] = 1.0
    _conflict_occ_cache[key] = (occ_now, occ_fut)
    if len(_conflict_occ_cache) > 64:
        _conflict_occ_cache.clear()
        _conflict_occ_cache[key] = (occ_now, occ_fut)
    return occ_now, occ_fut


def conflict_move_inflate(pos, pitch, timestamp, moving_obstacles, self_name) -> float:
    """Minimal CS-PIBT-style online hook: inflate only first 1–2 steps of each A*."""
    global _conflict_search_calls
    if not CONFLICT_PRED:
        return 1.0
    # Scope: only near search root (local shield), avoid full-horizon cost explosion
    if _conflict_t0 is not None and int(timestamp) > int(_conflict_t0) + 1:
        return 1.0
    if _conflict_search_calls >= _CONFLICT_MAX_CALLS:
        return 1.0
    cp = _lazy_conflict()
    if cp is None:
        return 1.0
    x, y = int(pos[0]), int(pos[1])
    cache_key = (x, y, int(pitch) % 360, int(timestamp), self_name)
    if cache_key in _conflict_inflate_cache:
        return _conflict_inflate_cache[cache_key]
    r = _CONFLICT_LOCAL_R
    near = False
    for name, traj in moving_obstacles.items():
        if name == self_name or not traj:
            continue
        for dt in (0, 1, 2):
            t = timestamp + dt
            p = traj[t][:2] if t < len(traj) else traj[-1][:2]
            if abs(p[0] - x) <= r and abs(p[1] - y) <= r:
                near = True
                break
        if near:
            break
    if not near:
        _conflict_inflate_cache[cache_key] = 1.0
        return 1.0
    _conflict_search_calls += 1
    occ_now, occ_fut = _conflict_occ_pair(timestamp, moving_obstacles, self_name, x, y)
    risk = cp.risk_at(x, y, int(pitch), occ_now, occ_fut)
    out = max(1.0, CONFLICT_INFLATE) if risk >= CONFLICT_THRESH else 1.0
    _conflict_inflate_cache[cache_key] = out
    if len(_conflict_inflate_cache) > 4096:
        _conflict_inflate_cache.clear()
    return out

def get_agv_state(agv_list):
    agv_states = {
        agv["id"]: {
            # "pos": tuple(agv["pose"]),
            # "pitch": agv["pitch"],
            # "time": 0,
            "state": agv["pose"] + (0,agv["pitch"]),
            "task_id": None,
            "path": [],
            "load_point": None,
            "end_point": None,
            "priority": False
        } for agv in agv_list
    }
    # print(agv_states)
    return agv_states

def get_end_points(end_point_name,end_point):
    # Dropoff 13 = Beijing (6,4). Its west pad (5,4) is the west trunk —
    # unloading there blocks every other AGV. Only north and east pads.
    if str(end_point_name) == "Beijing":
        possible_pos = [
            (1, 0),   # 右
            (0, 1),   # 上
        ]
    else:
        possible_pos = [
            (1, 0),    # 右
            (-1, 0), # 左
            (0, 1),   # 上
            (0, -1)  # 下
        ]
    # print(end_point)
    # print(end_point_name)
    unload_point = end_point[end_point_name]
    possible_end_points = []
    for x,y in possible_pos:
        possible_end_points.append((x+unload_point[0],y+unload_point[1]))
    # print(possible_end_point)
    return unload_point,possible_end_points

def get_pickup_coord(start_point_name, original_coord):
    if start_point_name in ["Tiger", "Dragon", "Horse"]:
        return (original_coord[0] + 1, original_coord[1])
    else:
        return (original_coord[0] - 1, original_coord[1])
    
def get_task_list(agv_task_path):
    all_tasks = {}
    with open(agv_task_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["start_point"] not in all_tasks:
                all_tasks[row["start_point"]] = []
            all_tasks[row["start_point"]].append({
                "task_id": row["task_id"],
                "start_point": row["start_point"].strip(),
                "end_point": row["end_point"].strip(),
                "priority": row["priority"],
                "remaining_time": int(row["remaining_time"]) if row["remaining_time"] not in [None, "", "None"] else None
            })
    return all_tasks

def get_task_state(start_points,all_tasks,all_end_points):
    task_states = {}
    for task_name,task_dict in all_tasks.items():
        if task_name not in task_states:
            task_states[task_name] = []
        # print(task_name,task_dict)
        for i in range(len(task_dict)):
            pickup_pos = get_pickup_coord(task_name,start_points[task_name])
            unload_point,end_points = get_end_points(task_dict[i]["end_point"],all_end_points)
            if task_dict[i]["priority"] == "Urgent":
                # task_states[task_name]["remaining_time"] = task_dict[i]["remaining_time"]
                task_states[task_name].append({
                "task_id": task_dict[i]["task_id"],
                "pickup_point": pickup_pos,
                "unload_point": unload_point,
                "end_points": end_points,
                "destination": task_dict[i]["end_point"],
                "priority": task_dict[i]["priority"],
                "numbers_before_urgent": 0,
                # "remaining_time": 300,
                "numbers_left": len(all_tasks[task_name])-i-1
            })
                for j in range(i):
                    task_states[task_name][j]['numbers_before_urgent'] = i-j
            else:
                task_states[task_name].append({
                "task_id": task_dict[i]["task_id"],
                "pickup_point": pickup_pos,
                "unload_point": unload_point,
                "end_points": end_points,
                "destination": task_dict[i]["end_point"],
                "priority": task_dict[i]["priority"],
                "numbers_before_urgent": -1,
                # "remaining_time": 300,
                "numbers_left": len(all_tasks[task_name])-i-1,
            })
    
    return task_states

def get_object_position(agv_position_path):
    start_points, end_points, agv_list = {}, {}, []
    with open(agv_position_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t, name = row["type"].strip(), row["name"].strip()
            x, y = int(row["x"]), int(row["y"])
            if t == "start_point":
                start_points[name] = (x, y)
            elif t == "end_point":
                end_points[name] = (x, y)
            elif t == "agv":
                agv_list.append({
                    "id": name,
                    "pose": (x, y),
                    "pitch": int(row["pitch"])
                })
    return start_points, end_points, agv_list


def append_to_csv(steps):

    with open(AGV_TRAJECTORY_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name", "X", "Y", "pitch", "loaded", "destination", "Emergency", "task-id"])
        for step in steps.values():
            for s in step:
                writer.writerow([
                    s["timestamp"],
                    s["name"],
                    s["X"],
                    s["Y"],
                    s["pitch"],
                    str(s["loaded"]).lower(),
                    s["destination"],
                    str(s["Emergency"]).lower(),
                    s.get("task-id", "")

                ])
              
              
"""检查轨迹文件中是否存在碰撞或对穿情况"""
def check_trajectory_conflicts(trajectory_file):
    # 读取轨迹文件
    agv_positions = {}  # 格式: {timestamp: {agv_name: (x, y)}}
    with open(trajectory_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['timestamp'] == 'timestamp':  # 跳过表头行
                continue
            timestamp = int(row['timestamp'])
            agv_name = row['name']
            x = int(row['X'])
            y = int(row['Y'])
            
            if timestamp not in agv_positions:
                agv_positions[timestamp] = {}
            agv_positions[timestamp][agv_name] = (x, y)
    
    has_conflict = False
    
    # 检查每个时间点的碰撞
    for t in sorted(agv_positions.keys()):
        # 检查同一时间点的位置碰撞
        positions = agv_positions[t]
        checked_agvs = set()
        
        for agv1, pos1 in positions.items():
            for agv2, pos2 in positions.items():
                if agv1 != agv2 and (agv1, agv2) not in checked_agvs and (agv2, agv1) not in checked_agvs:
                    checked_agvs.add((agv1, agv2))
                    
                    # 检查位置碰撞
                    if pos1 == pos2:
                        print(f"时间 {t}: {agv1} 和 {agv2} 在位置 {pos1} 发生碰撞")
                        has_conflict = True
                    
                    # 如果有下一个时间点，检查对穿
                    if t + 1 in agv_positions and agv1 in agv_positions[t+1] and agv2 in agv_positions[t+1]:
                        next_pos1 = agv_positions[t+1][agv1]
                        next_pos2 = agv_positions[t+1][agv2]
                    
                        
                        # 检查交叉路径
                        if pos1 == next_pos2 and pos2 == next_pos1:
                            print(f"时间 {t}-{t+1}: {agv1} 和 {agv2} 发生路径交叉")
                            print(f"  {agv1}: {pos1} -> {next_pos1}")
                            print(f"  {agv2}: {pos2} -> {next_pos2}")
                            has_conflict = True
    
    if not has_conflict:
        print("无对穿或碰撞")
    
    return has_conflict


def summarize_utilization(trajectory_file):
    """统计载货利用率：平均/最大同时 loaded 数，以及空闲等待占比。"""
    by_t = {}
    with open(trajectory_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['timestamp'] == 'timestamp':
                continue
            t = int(row['timestamp'])
            loaded = str(row.get('loaded', '')).lower() in ('true', '1', 'yes')
            by_t.setdefault(t, {'loaded': 0, 'total': 0})
            by_t[t]['total'] += 1
            if loaded:
                by_t[t]['loaded'] += 1
    if not by_t:
        print("利用率: 无数据")
        return
    loads = [v['loaded'] for v in by_t.values()]
    avg_loaded = sum(loads) / len(loads)
    max_loaded = max(loads)
    print(f"利用率: avg_loaded={avg_loaded:.2f}, max_loaded={max_loaded}, horizon={max(by_t)}")
              
  
class Simulation:
    def __init__(self, agv_states, task_states, env, dimension = DEFAULT_GRID_SIZE):
        self.agv_states = agv_states
        self.task_states = task_states
        self.dimension = dimension
        self.surface_tasks = {}
        self.tried_tasks = set()  # 改为 set 来记录所有失败的组合
        self.assign_cooldown = {}  # agv_name -> 最早可再派单的时刻
        self.time = 0
        self.agvs = []
        self.env = env
        self._pickup_committed = set()  # task_ids whose FIFO pickup already advanced queue
        self._assign_fail_streak = 0  # consecutive ticks with idle+surface but 0 assigns

        # Congestion / holding (safe default ON; disable via env or attribute)
        self.congestion_control = CONGESTION_CONTROL
        self.endpoint_capacity_extra = ENDPOINT_CAPACITY_EXTRA
        self.astar_max_path_len = ASTAR_MAX_PATH_LEN
        self.astar_max_visited = ASTAR_MAX_VISITED
        self.astar_max_frontier = ASTAR_MAX_FRONTIER
        self.astar_wall_budget_s = ASTAR_WALL_BUDGET_S
        # Neural A* (mirrors module flags; prefer set_neural_astar for runtime toggles)
        self.residual_heuristic = RESIDUAL_HEURISTIC
        self.residual_mode = RESIDUAL_MODE
        self.conflict_pred = CONFLICT_PRED
        
        #初始化12个AGV类
        self.init_agvs()
        # print(self.agvs)
        # print(self.agv_states)
        #初始化供货点当前任务
        self.init_surface_tasks()

        # self.a_star = A_star()

    def endpoint_pipeline_cap(self, end_points) -> int:
        """Max concurrent AGVs allowed toward these unload slots."""
        n_slots = max(1, len(end_points or []))
        if not getattr(self, "congestion_control", True):
            return max(n_slots, len(self.agvs))
        extra = int(getattr(self, "endpoint_capacity_extra", 1))
        return max(1, n_slots + max(0, extra))

    def endpoint_cost_multiplier(self, end_points) -> float:
        """Shared soft/hard endpoint congestion multiplier (M0 + patched allocators)."""
        if not end_points:
            return 1.0
        parked = self._count_endpoint_occupancy(end_points)
        active = self._count_active_unloaders(end_points)
        n_slots = max(1, len(end_points))
        hard_cap = self.endpoint_pipeline_cap(end_points)
        mult = 1.0
        if parked >= n_slots:
            mult *= 5.0
        if active >= hard_cap:
            return float("inf")
        if active >= n_slots:
            # Milder ramp than legacy; hard_cap already throttles Extreme/H03
            mult *= 1.0 + 0.08 * (active - n_slots + 1)
        return mult
    
    def init_agvs(self):
        for agv_name, agv_state in self.agv_states.items():
            agv = AGV(agv_name, agv_state['state'], None)
            # path 按时间索引：path[t] 为时刻 t 的状态
            agv.path = [agv_state['state']]
            agv.steps = [{
                "timestamp": 0,
                "name": agv_name,
                "X": agv_state['state'][0],
                "Y": agv_state['state'][1],
                "pitch": agv_state['state'][3],
                "loaded": "FALSE",
                "destination": "",
                "Emergency": "FALSE",
                "task-id": ""
            }]
            self.agvs.append(agv)
            self.env.moving_obstacles[agv_name] = [agv_state['state']]
    def init_surface_tasks(self):
        for name, task_info in self.task_states.items():
            if task_info: 
                first_task = task_info[0] 
                task_id = first_task['task_id'] 
                task_details = {k: v for k, v in first_task.items() if k != 'task_id'}
                if 'end_points' in task_details:
                    task_details['end_points'] = list(task_details['end_points'])
                self.surface_tasks[task_id] = task_details
                self.surface_tasks[task_id]['pickup_name'] = name
        print(f"初始化 surface_tasks: {list(self.surface_tasks.keys())}")
        
    def get_cost_matrix_and_allocate_task(self,unassigned_agvs):
        cost_matrix = {}
        assigned_task = []
        min_agv = None
        min_task_id = None
        min_cost = float('inf')
        # Surface multitask (promote-on-assign) may expose the next FIFO head while
        # an unloaded AGV still approaches the same station. Keep one pad approacher
        # per station — cross-station fill is the win, not same-station convoy.
        busy_stations = self._stations_with_unloaded_approacher()
        max_k = max(1, int(os.environ.get("AGV_STATION_MAX_UNLOADED", "1") or "1"))
        # print(f"当前 surface_tasks: {list(self.surface_tasks.keys())}")
        # print(f"当前 tried_tasks: {self.tried_tasks}")
        # print(self.surface_tasks)
        for agv in unassigned_agvs:
            cost_matrix[agv.name] = {}
            for task_id,task_info in self.surface_tasks.items():
                # 计算AGV到任务的距离
                cost = self.manhattan_distance(agv.state[:2], task_info['pickup_point']) 
                # 考虑紧急程度权重
                urgent_weight = 1.0
                if task_info['numbers_before_urgent'] >= 0:
                    urgent_weight = 0.7
                cost *= urgent_weight

                # 端点容量：拥堵控制开启时 hard_cap≈n_slots+extra，避免全车涌入同卸货点
                end_pts = task_info.get('end_points', [])
                if end_pts:
                    cost *= self.endpoint_cost_multiplier(end_pts)

                if (agv.name, task_id) in self.tried_tasks:
                    cost = float('inf')
                if self.assign_cooldown.get(agv.name, -1) > self.time:
                    cost = float('inf')
                st = str(task_info.get("pickup_name") or "")
                if st and int(busy_stations.get(st, 0)) >= max_k:
                    cost = float('inf')
                    
                cost_matrix[agv.name][task_id] = cost
                
                if cost < min_cost:
                    min_agv = agv
                    min_task_id = task_id
                    min_cost = cost
                    
        if min_agv and min_task_id and min_cost != float('inf'):
            # 记录分配结果（end_points 必须拷贝，避免 A* 原地 remove 污染后续任务）
            # Coerce to int tuples: list vs tuple breaks A* goal checks / empty paths.
            info = self.surface_tasks[min_task_id]
            pk = info['pickup_point']
            ends = info.get('end_points') or []
            assigned_task = {
                "agv": min_agv.name,
                "task_id": min_task_id,
                "agv_start_point": min_agv.state,
                "pickup_point": (int(pk[0]), int(pk[1])),
                "end_points": [(int(e[0]), int(e[1])) for e in ends if e is not None and len(e) >= 2],
                "destination": info['destination'],
                "priority": info['priority'],
                "pickup_name": info['pickup_name'],
            }
            print(f"分配: AGV {min_agv.name} → 任务 {min_task_id}, 代价: {min_cost}")
        else:
            print("没有可用的任务分配")
            
        return assigned_task

    def _escape_task_prefixes(self):
        return ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")

    def _stations_with_unloaded_approacher(self):
        """Count unloaded AGVs still heading to each pickup station."""
        counts = {}
        inflight = getattr(self, "_inflight_tasks", None) or {}
        prefixes = self._escape_task_prefixes()
        for agv in self.agvs:
            tid = getattr(agv, "task_id", None)
            if tid is None:
                continue
            s = str(tid)
            if s.startswith(prefixes):
                continue
            step = self._step_at_time(agv, int(self.time))
            if step is not None:
                loaded = str(step.get("loaded", "")).lower() in ("true", "1", "yes")
                if loaded:
                    continue
            info = inflight.get(agv.name) or {}
            st = str(info.get("pickup_name") or self._pickup_name_for_task(tid) or "")
            if not st:
                continue
            counts[st] = int(counts.get(st, 0)) + 1
        return counts

    def _reserved_task_ids(self):
        owned = set()
        prefixes = self._escape_task_prefixes()
        for agv in self.agvs:
            tid = getattr(agv, "task_id", None)
            if tid is None:
                continue
            s = str(tid)
            if s.startswith(prefixes):
                continue
            owned.add(s)
        committed = getattr(self, "_pickup_committed", None) or set()
        owned |= {str(x) for x in committed}
        return owned

    def get_unassigned_agvs(self):
        unassigned_agvs = []
        for agv in self.agvs:
            # 避让/逃离中的 AGV 暂不接新任务
            if agv.task_id is not None:
                continue
            # Still physically loaded (steps) → refuse new MAPD assign even if
            # task_id was cleared early (historical unload false-positive).
            step = self._step_at_time(agv, int(self.time))
            if step is not None:
                loaded = str(step.get("loaded", "")).lower() in ("true", "1", "yes")
                tid = str(step.get("task-id") or "").strip()
                if loaded and tid and not tid.startswith(
                    ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
                ):
                    continue
            unassigned_agvs.append(agv)
        return unassigned_agvs

    def _count_endpoint_occupancy(self, end_points):
        """统计路径终点落在这些卸货点上的 AGV 数量。"""
        ep_set = set(end_points)
        return sum(1 for agv in self.agvs if agv.path and agv.path[-1][:2] in ep_set)

    def _count_active_unloaders(self, end_points):
        """统计仍将访问这些卸货点的忙碌 AGV（流水线在途数）。"""
        ep_set = set(end_points)
        count = 0
        for agv in self.agvs:
            if agv.task_id is None or str(agv.task_id).startswith('escape_'):
                continue
            future = agv.path[self.time:] if agv.path else []
            if any(p[:2] in ep_set for p in future):
                count += 1
        return count

    def get_all_unload_points(self):
        points = set()
        dps = self.env.destination_points
        coords = list(dps.values()) if isinstance(dps, dict) else list(dps or [])
        for dest in coords:
            if dest is None or len(dest) < 2:
                continue
            x, y = int(dest[0]), int(dest[1])
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                points.add((x + dx, y + dy))
        return points

    def get_needed_unload_points(self):
        needed = set()
        for task in self.surface_tasks.values():
            needed.update(task.get('end_points', []))
        return needed

    def get_safe_parking_points(self):
        return []

    def _unload_pad_set(self) -> set:
        """All known unload pads (dest rings + surface end_points)."""
        pads = set(self.get_all_unload_points())
        for task in (self.surface_tasks or {}).values():
            for e in task.get("end_points") or []:
                if e is not None and len(e) >= 2:
                    pads.add((int(e[0]), int(e[1])))
        return pads

    def evacuate_hub_idles_far(self, min_r: int = 4, max_r: int = 8) -> bool:
        """Force idle AGVs within manh≤2 of any unload pad to far holding.

        Does not require the hub pipeline to be full — used on assign-fail
        melt-down and before M0→wave handoff so pad-blocking idles clear.
        """
        pads = self._unload_pad_set()
        if not pads:
            return False
        occupied, static_obs = self._occupied_cells_now()
        forbidden = set(pads)
        moved = False
        for agv in list(self.agvs):
            tid = agv.task_id
            if tid is not None and not str(tid).startswith("escape_"):
                continue
            # escape_* counts as relocatable idle for pad clearing
            if tid is not None and str(tid).startswith("escape_"):
                # already relocating; skip
                continue
            pos = agv.state[:2]
            near = any(
                abs(pos[0] - ep[0]) + abs(pos[1] - ep[1]) <= 2 for ep in pads
            )
            on_pad = pos in pads
            if not (near or on_pad):
                continue
            goal = self._pick_holding_cell(
                pos, forbidden, occupied, static_obs, min_r=int(min_r), max_r=int(max_r)
            )
            if goal is None:
                continue
            if self._relocate_idle_agv(agv, goal, occupied):
                moved = True
        return moved

    def scatter_idle_agvs(self, min_r: int = 4, max_r: int = 14) -> int:
        """Relocate ALL true-idle AGVs to spread holding cells.

        SH02 plateau@165: 12 idle AGVs clustered → assign A* all-inf/fail while
        surface>0. Pad-only evacuate skipped them (not near unload pads).
        """
        occupied, static_obs = self._occupied_cells_now()
        pads = self._unload_pad_set()
        forbidden = set(pads)
        moved = 0
        prefixes = self._escape_task_prefixes()
        t_now = int(self.time)
        for agv in list(self.agvs):
            tid = getattr(agv, "task_id", None)
            if tid is not None and not str(tid).startswith(prefixes):
                continue
            # Do not interrupt mid-flight escape (r11 re-scatter thrash).
            if tid is not None and str(tid).startswith(prefixes):
                path = getattr(agv, "path", None) or []
                if path and t_now < len(path) - 1:
                    continue
                agv.task_id = None
            pos = tuple(agv.state[:2])
            goal = self._pick_holding_cell(
                pos, forbidden, occupied, static_obs, min_r=int(min_r), max_r=int(max_r)
            )
            if goal is None:
                # try farther ring
                goal = self._pick_holding_cell(
                    pos, forbidden, occupied, static_obs, min_r=int(max_r), max_r=int(max_r) + 6
                )
            if goal is None:
                continue
            if self._relocate_idle_agv(agv, goal, occupied):
                moved += 1
        return int(moved)

    def _occupied_cells_now(self):
        static_obs = set(self.env.get_static_obstacles())
        occupied = set(static_obs)
        for path in self.env.moving_obstacles.values():
            if path:
                occupied.add(path[-1][:2])
        for agv in self.agvs:
            if agv.path and self.time < len(agv.path):
                occupied.add(agv.path[self.time][:2])
            elif agv.state:
                occupied.add(agv.state[:2])
        return occupied, static_obs

    def _pick_holding_cell(self, pos, forbidden, occupied, static_obs, min_r=2, max_r=4):
        """Nearest free holding cell around pos (not on unload slots / static)."""
        holding = []
        for radius in range(min_r, max_r + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) != radius:
                        continue
                    cand = (pos[0] + dx, pos[1] + dy)
                    if not _in_playable(cand[0], cand[1]):
                        continue
                    if cand in static_obs or cand in forbidden or cand in occupied:
                        continue
                    holding.append(cand)
            if holding:
                break
        holding.sort(key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))
        return holding[0] if holding else None

    def _relocate_idle_agv(self, agv, best_goal, occupied):
        escape_task = {
            "agv": agv.name,
            "task_id": f"escape_{agv.name}",
            "destination": "",
            "priority": "Normal",
        }
        path, _steps = plan_relocation(
            escape_task, agv.state, best_goal,
            self.env.get_static_obstacles(), self.env.moving_obstacles,
            wall_budget_s=getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S),
            max_visited=getattr(self, "astar_max_visited", ASTAR_MAX_VISITED),
        )
        if not path:
            return False
        print(f"AGV {agv.name} holding/让位 {agv.state[:2]} -> {best_goal}, 步数 {len(path)}")
        agv.task_id = escape_task["task_id"]
        agv.priority = False
        kin = self._append_kinematic_path(
            agv, path, escape_task, loaded=False, check_ends=[tuple(best_goal)]
        )
        if not kin:
            return False
        self.update_env(agv.name, kin)
        occupied.add(best_goal)
        return True

    def evacuate_blocking_idle_agvs(self):
        """If idle AGVs sit on/near needed unload pads, force far holding."""
        needed = self.get_needed_unload_points()
        if not needed:
            return False
        for agv in self.agvs:
            tid = agv.task_id
            if tid is not None and not str(tid).startswith("escape_"):
                continue
            pos = agv.state[:2]
            if pos in needed or any(
                abs(pos[0] - ep[0]) + abs(pos[1] - ep[1]) <= 2 for ep in needed
            ):
                return self.evacuate_hub_idles_far(min_r=4, max_r=8)
        return False

    def hold_excess_near_congested_hubs(self):
        """拥堵开启时：空闲车在/紧贴已满配额卸货点 → 远 holding，避免堵口/干扰 A*。"""
        if not getattr(self, "congestion_control", True):
            return False

        congested = set()
        seen = set()
        # Surface hubs
        for task in self.surface_tasks.values():
            eps = tuple(sorted(task.get("end_points", [])))
            if not eps or eps in seen:
                continue
            seen.add(eps)
            if self._count_active_unloaders(list(eps)) >= self.endpoint_pipeline_cap(list(eps)):
                congested.update(eps)
        # Destination rings (covers in-flight when surface already cleared for that hub)
        dps = self.env.destination_points
        coords = list(dps.values()) if isinstance(dps, dict) else list(dps or [])
        for dest in coords:
            if dest is None or len(dest) < 2:
                continue
            x, y = int(dest[0]), int(dest[1])
            eps = (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            )
            if eps in seen:
                continue
            seen.add(eps)
            if self._count_active_unloaders(list(eps)) >= self.endpoint_pipeline_cap(list(eps)):
                congested.update(eps)

        if not congested:
            return False
        occupied, static_obs = self._occupied_cells_now()
        moved = False
        for agv in list(self.agvs):
            if agv.task_id is not None:
                continue
            pos = agv.state[:2]
            on_pad = pos in congested
            near = any(
                abs(pos[0] - ep[0]) + abs(pos[1] - ep[1]) <= 2 for ep in congested
            )
            # On pad OR within manh≤2 → must relocate (previously skipped on-pad).
            if not (on_pad or near):
                continue
            goal = self._pick_holding_cell(
                pos, congested, occupied, static_obs, min_r=4, max_r=8
            )
            if goal is None:
                continue
            if self._relocate_idle_agv(agv, goal, occupied):
                moved = True
        return moved
    
    def _try_assign_tasks(self, fail_budget, max_evacuate):
        """尽量把空闲车派满；路径失败则换 AGV/任务，必要时疏散堵口空闲车。

        Returns number of successful assigns this call.
        """
        evacuated = 0
        failed_agvs = set()
        successes = 0
        # Congestion on: fewer futile A* retries when hubs are saturated
        if getattr(self, "congestion_control", True):
            fail_budget = min(fail_budget, max(4, len(self.agvs) // 2 + 2))
        cap = int(os.environ.get("AGV_ASSIGN_FAIL_BUDGET", "0") or 0)
        if cap > 0:
            fail_budget = min(int(fail_budget), cap)
        while fail_budget > 0:
            unassigned_agvs = [a for a in self.get_unassigned_agvs() if a.name not in failed_agvs]
            if not unassigned_agvs or not self.surface_tasks:
                break

            assigned_task = self.get_cost_matrix_and_allocate_task(unassigned_agvs)
            if not assigned_task:
                # 还有未失败的空闲车但代价全 inf：尝试疏散；否则结束本轮
                if evacuated < max_evacuate and self.evacuate_blocking_idle_agvs():
                    evacuated += 1
                    self.tried_tasks.clear()
                    # 不清空 failed_agvs，避免同车反复空转 A*
                    continue
                if getattr(self, "congestion_control", True):
                    self.hold_excess_near_congested_hubs()
                break

            end_points = list(assigned_task['end_points'])
            path, steps = A_Star(
                assigned_task,
                assigned_task['agv_start_point'],
                assigned_task['pickup_point'],
                end_points,
                self.env.get_static_obstacles(),
                self.env.moving_obstacles,
                max_path_len=getattr(self, "astar_max_path_len", ASTAR_MAX_PATH_LEN),
                max_visited=getattr(self, "astar_max_visited", ASTAR_MAX_VISITED),
                max_frontier=getattr(self, "astar_max_frontier", ASTAR_MAX_FRONTIER),
                wall_budget_s=getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S),
            )
            # Pickup-then-delivery: assign path may stop at pickup (no dropoff yet).
            check_ends = end_points
            if (
                path
                and str(os.environ.get("AGV_PICKUP_THEN_DELIVERY", "")).strip().lower()
                in ("1", "true", "yes", "on")
                and not any(p[:2] in end_points for p in path)
            ):
                check_ends = [assigned_task["pickup_point"]]
            if path and self.env.is_valid_path(check_ends, path, self.time, assigned_task['agv']):
                # Commit only after kinematic expand re-validates spacetime.
                # Expand inserts turn ticks / retimes → occupancy ≠ A* path.
                kin = self.update_agvs(
                    assigned_task, path, steps, check_ends=check_ends
                )
                if kin:
                    self.reserve_assigned_task(assigned_task['task_id'])
                    self.update_env(assigned_task['agv'], kin)
                    self.update_tried_tasks(assigned_task['agv'])
                    successes += 1
                    continue
                # A* path OK but expanded kin collides → planning failure.

            # PIBT hot handoff is opt-in only. Default = parking / evacuate / RulePark
            # (A* fail → cooldown + hub evacuate; RulePark parks on assign-fail streak).
            pibt_on = str(os.environ.get("AGV_PIBT_FALLBACK", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if pibt_on:
                try:
                    from ml_research.benchmarks.pibt_fallback import plan_pibt_assign

                    t_p0 = _time.perf_counter()
                    pibt_path = plan_pibt_assign(
                        ego=str(assigned_task["agv"]),
                        start=tuple(assigned_task["agv_start_point"]),
                        pickup=tuple(assigned_task["pickup_point"]),
                        end_points=list(end_points),
                        static_obstacles=self.env.get_static_obstacles(),
                        moving_obstacles=self.env.moving_obstacles,
                        all_names=[a.name for a in self.agvs],
                        max_steps=int(os.environ.get("AGV_PIBT_MAX_STEPS", "160")),
                    )
                    pibt_ms = (_time.perf_counter() - t_p0) * 1000.0
                    if pibt_path and self.env.is_valid_path(
                        end_points, pibt_path, self.time, assigned_task["agv"]
                    ):
                        kin = self.update_agvs(
                            assigned_task, pibt_path, [], check_ends=end_points
                        )
                        if kin:
                            print(
                                f"[PIBT] handoff AGV {assigned_task['agv']} "
                                f"task {assigned_task['task_id']} "
                                f"len={len(pibt_path)} wall={pibt_ms:.1f}ms",
                                flush=True,
                            )
                            self.reserve_assigned_task(assigned_task["task_id"])
                            self.update_env(assigned_task["agv"], kin)
                            self.update_tried_tasks(assigned_task["agv"])
                            successes += 1
                            continue
                    print(
                        f"[PIBT] handoff failed AGV {assigned_task['agv']} "
                        f"task {assigned_task['task_id']} wall={pibt_ms:.1f}ms",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[PIBT] handoff error: {exc}", flush=True)

            fail_budget -= 1
            if str(os.environ.get("AGV_QUIET_ASSIGN", "1")).strip().lower() not in (
                "1",
                "true",
                "yes",
                "on",
            ):
                print(
                    f"AGV {assigned_task['agv']} 无法为{assigned_task['task_id']}任务找到有效路径"
                )
            pair = (assigned_task['agv'], assigned_task['task_id'])
            self.tried_tasks.add(pair)
            ttl = int(os.environ.get("AGV_TRIED_TTL", "5") or 0)
            if ttl > 0:
                until = getattr(self, "_tried_until", None)
                if not isinstance(until, dict):
                    until = {}
                    self._tried_until = until
                until[pair] = int(self.time) + ttl
            # 仅当该车对当前所有 surface 任务都试过失败，才冷却；否则换其他任务继续试
            agv_name = assigned_task['agv']
            remaining = [
                tid for tid in self.surface_tasks
                if (agv_name, tid) not in self.tried_tasks
            ]
            if not remaining:
                failed_agvs.add(agv_name)
                self.assign_cooldown[agv_name] = self.time + 2
            if evacuated < max_evacuate and self._endpoints_fully_blocked(end_points):
                if self.evacuate_blocking_idle_agvs():
                    evacuated += 1
        return int(successes)

    def _expire_tried_tasks(self) -> None:
        ttl = int(os.environ.get("AGV_TRIED_TTL", "5") or 0)
        if ttl <= 0:
            self.tried_tasks.clear()
            self._tried_until = {}
            return
        until = getattr(self, "_tried_until", None)
        if not isinstance(until, dict):
            until = {}
            self._tried_until = until
        t = int(self.time)
        keep = set()
        for pair in self.tried_tasks:
            if int(until.get(pair, t)) >= t:
                keep.add(pair)
        self.tried_tasks = keep

    def time_forward(self):
        self.time += 1
        self._expire_tried_tasks()

        # Recover orphaned FIFO heads before assign (empty surface + idle fleet).
        self.heal_orphaned_surface_tasks()
        # Path ended but real task unfinished → continue A* or release reservation
        # (otherwise last task can hang forever with empty surface).
        self._continue_or_release_stale_path_end_tasks()
        if self._get_mo_horizon() > 0:
            self._replan_horizon_continuations()

        if getattr(self, "congestion_control", True):
            self.evacuate_blocking_idle_agvs()
            self.hold_excess_near_congested_hubs()

        hook = getattr(self, "_pre_assign_hook", None)
        if callable(hook):
            try:
                hook(self)
            except Exception:  # noqa: BLE001
                pass

        had_surface = bool(self.surface_tasks)
        had_idle = bool(self.get_unassigned_agvs())

        boost_every = int(os.environ.get("AGV_ASTAR_BOOST_EVERY", "8") or 0)
        boost_s = float(os.environ.get("AGV_ASTAR_BOOST_S", "0.40") or 0.0)
        saved_budget = float(getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S))
        if boost_every > 0 and boost_s > 0 and int(self.time) % boost_every == 0:
            self.astar_wall_budget_s = max(saved_budget, boost_s)

        # 先派任务，尽量让空闲车投入（同目的地硬上限≈槽位数+extra）
        n_ok = int(
            self._try_assign_tasks(fail_budget=max(12, len(self.agvs)), max_evacuate=2)
            or 0
        )
        self.astar_wall_budget_s = saved_budget
        self.update_agv_state()
        # Second pass only after a success (just-finished unload). Retrying
        # a full fail_budget of timed-out A* every tick costs seconds/tick.
        if n_ok > 0 and self.surface_tasks and self.get_unassigned_agvs():
            n_ok += int(
                self._try_assign_tasks(fail_budget=max(3, len(self.agvs) // 2), max_evacuate=1)
                or 0
            )
            if getattr(self, "congestion_control", True):
                self.hold_excess_near_congested_hubs()

        if had_surface and had_idle:
            if n_ok <= 0:
                self._assign_fail_streak = int(getattr(self, "_assign_fail_streak", 0) or 0) + 1
            else:
                self._assign_fail_streak = 0
                # Restore full A* wall budget after a successful assign (clamp
                # at streak>=8 must not stick forever and starve late assigns).
                self.astar_wall_budget_s = float(ASTAR_WALL_BUDGET_S)
            if int(self._assign_fail_streak) >= 8:
                cur_budget = float(
                    getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S)
                )
                # Boost (not clamp-to-0.15). Only full evacuate/clear every 8
                # fail ticks — doing it every tick burned ~9s wall/tick on SH02.
                if int(self._assign_fail_streak) % 8 == 0:
                    self.astar_wall_budget_s = max(cur_budget, 0.45)
                    self.tried_tasks.clear()
                    self._tried_until = {}
                    self.evacuate_hub_idles_far(min_r=4, max_r=8)
                else:
                    self.astar_wall_budget_s = max(cur_budget, 0.30)
            # Rule park/artery recovery on assign failure (throttled).
            if (
                int(self._assign_fail_streak) >= 1
                and getattr(self, "_traffic_recovery", None) is not None
                and int(self.time) % 5 == 0
            ):
                try:
                    from ml_research.benchmarks.lane_traffic_ai.recovery import (
                        maybe_trigger_from_sim,
                    )

                    maybe_trigger_from_sim(self)
                    ctrl = self._traffic_recovery
                    if ctrl is not None:
                        ctrl.tick(self)
                except Exception as exc:  # noqa: BLE001
                    print(f"[TRAFFIC] recovery tick failed: {exc}", flush=True)
        elif n_ok > 0:
            self._assign_fail_streak = 0
            self.astar_wall_budget_s = float(ASTAR_WALL_BUDGET_S)
        # Always advance recovery agents that are already registered
        ctrl = getattr(self, "_traffic_recovery", None)
        if ctrl is not None and getattr(ctrl, "agents", None):
            try:
                ctrl.tick(self)
            except Exception:  # noqa: BLE001
                pass

    def _endpoints_fully_blocked(self, end_points):
        if not end_points:
            return True
        occupied = set()
        for path in self.env.moving_obstacles.values():
            if path:
                occupied.add(path[-1][:2])
        return all(ep in occupied for ep in end_points)

    def _has_allocatable_pair(self, unassigned_agvs):
        for agv in unassigned_agvs:
            for task_id in self.surface_tasks:
                if (agv.name, task_id) not in self.tried_tasks:
                    return True
        return False
        
            
            
            
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def update_task_states(self, name):
        if len(self.task_states[name]) == 1:
            del self.task_states[name]
        else:
            self.task_states[name] = self.task_states[name][1:]

    def reserve_assigned_task(self, task_id):
        """分配成功：从 surface 移除；可选立刻露出下一未预留单（surface 多任务）。"""
        pickup_name = None
        if task_id in self.surface_tasks:
            pickup_name = self.surface_tasks[task_id].get("pickup_name")
            del self.surface_tasks[task_id]
        promote = str(os.environ.get("AGV_PROMOTE_ON_ASSIGN", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if promote:
            print(f"预留任务 {task_id}（派单 promote 下一未预留单）")
            if not pickup_name:
                pickup_name = self._pickup_name_for_task(task_id)
            if pickup_name:
                self.promote_next_unreserved(pickup_name)
        else:
            print(f"预留任务 {task_id}（取货前不推进队列）")

    def promote_next_unreserved(self, name, quiet: bool = False) -> bool:
        """Expose the next queue task that is not already reserved/owned."""
        if name not in (self.task_states or {}) or not self.task_states[name]:
            return False
        reserved = self._reserved_task_ids()
        for item in self.task_states[name]:
            next_id = item.get("task_id")
            if next_id is None:
                continue
            sid = str(next_id)
            if sid in reserved:
                continue
            if next_id in self.surface_tasks or sid in self.surface_tasks:
                return False
            task_details = {k: v for k, v in item.items() if k != "task_id"}
            if "end_points" in task_details:
                task_details["end_points"] = list(task_details["end_points"])
            self.surface_tasks[next_id] = task_details
            self.surface_tasks[next_id]["pickup_name"] = name
            if not quiet:
                print(f"派单/恢复推进 {name} → 表面任务 {next_id}")
            return True
        return False

    def promote_station_after_pickup(self, name):
        """取货完成后：将站台队列下一未预留单暴露到 surface。"""
        self.promote_next_unreserved(name)

    def heal_orphaned_surface_tasks(self) -> int:
        """Re-expose station queue heads that are reserved but no AGV still owns them.

        Assign removes the head from ``surface_tasks`` before pickup. If the AGV
        later drops ``task_id`` (escape overwrite, cancel without restore, path
        end) without FIFO pickup, the station is stuck: work remains in
        ``task_states`` but ``surface_tasks`` stays empty → lifelong hang.
        """
        owned = set()
        for agv in self.agvs:
            tid = agv.task_id
            if tid is None:
                continue
            s = str(tid)
            if s.startswith(("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")):
                continue
            owned.add(s)
        restored = 0
        for name, queue in list((self.task_states or {}).items()):
            if not queue:
                continue
            head = queue[0]
            tid = str(head.get("task_id") or "")
            if not tid or tid in owned or tid in self.surface_tasks:
                continue
            if tid in getattr(self, "_pickup_committed", set()):
                # Should have been popped; skip rather than double-expose.
                continue
            task_details = {k: v for k, v in head.items() if k != "task_id"}
            if "end_points" in task_details:
                task_details["end_points"] = list(task_details["end_points"])
            self.surface_tasks[tid] = task_details
            self.surface_tasks[tid]["pickup_name"] = name
            restored += 1
            print(f"[HEAL] restore orphan surface {tid} @ {name}", flush=True)
        return restored

    def _continue_or_release_stale_path_end_tasks(self) -> int:
        """When an AGV sits at path-end with an unfinished real task, continue or release.

        Classic hang: last task reserved (off surface), path ends without unload,
        heal_orphaned skips because AGV still owns task_id → sim never finishes.
        """
        try:
            idle_try = max(1, int(os.environ.get("AGV_STALE_PATH_IDLE", "5") or "5"))
        except Exception:  # noqa: BLE001
            idle_try = 5
        try:
            release_after = max(
                idle_try + 1,
                int(os.environ.get("AGV_STALE_PATH_RELEASE", "50") or "50"),
            )
        except Exception:  # noqa: BLE001
            release_after = 50
        cool = getattr(self, "_stale_path_replan_at", None)
        if not isinstance(cool, dict):
            cool = {}
            self._stale_path_replan_at = cool
        fixed = 0
        t_now = int(self.time)
        for agv in self.agvs:
            tid = getattr(agv, "task_id", None)
            if tid is None:
                continue
            tid_s = str(tid)
            if tid_s.startswith(
                ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
            ):
                continue
            if not agv.path:
                continue
            path_end_t = len(agv.path) - 1
            if t_now < path_end_t:
                continue
            idle = t_now - path_end_t
            if idle < idle_try:
                continue
            if self._has_completed_unload_for_task(agv, tid_s):
                continue

            step = self._step_at_time(agv, t_now)
            step_loaded = bool(
                step is not None
                and str(step.get("loaded", "")).lower() in ("true", "1", "yes")
            )
            # Rising-edge / true FIFO pickup only — ignore path-end fake loaded.
            picked = tid_s in (getattr(self, "_pickup_committed", None) or set())
            really_carrying = bool(step_loaded and picked)

            # Never release cargo already taken from the station queue.
            if really_carrying or picked:
                next_ok_t = int(cool.get(agv.name, 0) or 0)
                if t_now < next_ok_t:
                    continue
                cool[agv.name] = t_now + 8
                info = self._task_info_for_agv(agv) or self._lookup_task_info_by_id(
                    tid_s
                )
                if not info:
                    continue
                end_points = list(info.get("end_points") or [])
                if not end_points:
                    continue
                start = agv.path[min(t_now, len(agv.path) - 1)]
                start_xy = (int(start[0]), int(start[1]), int(start[3]) % 360)
                cur_xy = (int(start[0]), int(start[1]))
                pads = {
                    (int(e[0]), int(e[1]))
                    for e in end_points
                    if e is not None and len(e) >= 2
                }
                if cur_xy in pads:
                    continue
                task = {
                    "agv": agv.name,
                    "task_id": tid_s,
                    "destination": info.get("destination", ""),
                    "priority": info.get("priority", "Normal"),
                    "end_points": end_points,
                    "force_full_path": True,
                }
                path, steps = A_Star(
                    task,
                    start_xy,
                    cur_xy,
                    end_points,
                    self.env.get_static_obstacles(),
                    self.env.moving_obstacles,
                    max_path_len=getattr(self, "astar_max_path_len", ASTAR_MAX_PATH_LEN),
                    max_visited=getattr(self, "astar_max_visited", ASTAR_MAX_VISITED),
                    max_frontier=getattr(self, "astar_max_frontier", ASTAR_MAX_FRONTIER),
                    wall_budget_s=getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S),
                )
                if not path or not self.env.is_valid_path(
                    end_points, path, self.time, agv.name
                ):
                    continue
                kin = self._append_kinematic_path(
                    agv,
                    path,
                    task,
                    loaded=True,
                    pickup_point=None,
                    end_points=end_points,
                    src_steps=steps,
                    check_ends=end_points,
                )
                if kin:
                    self.update_env(agv.name, kin)
                    fixed += 1
                    print(
                        f"[HEAL] continue delivery {agv.name} {tid_s} "
                        f"@t={t_now} idle={idle}",
                        flush=True,
                    )
                continue

            # Not picked up: try finish pickup+delivery; else release reservation.
            next_ok_t = int(cool.get(agv.name, 0) or 0)
            if t_now >= next_ok_t:
                cool[agv.name] = t_now + 8
                info = self._task_info_for_agv(agv) or self._lookup_task_info_by_id(
                    tid_s
                )
                pickup = (info or {}).get("pickup_point")
                end_points = list((info or {}).get("end_points") or [])
                if (
                    info
                    and pickup is not None
                    and len(pickup) >= 2
                    and end_points
                ):
                    start = agv.path[min(t_now, len(agv.path) - 1)]
                    start_xy = (int(start[0]), int(start[1]), int(start[3]) % 360)
                    pk = (int(pickup[0]), int(pickup[1]))
                    task = {
                        "agv": agv.name,
                        "task_id": tid_s,
                        "destination": info.get("destination", ""),
                        "priority": info.get("priority", "Normal"),
                        "end_points": end_points,
                        "force_full_path": True,
                    }
                    path, steps = A_Star(
                        task,
                        start_xy,
                        pk,
                        end_points,
                        self.env.get_static_obstacles(),
                        self.env.moving_obstacles,
                        max_path_len=getattr(
                            self, "astar_max_path_len", ASTAR_MAX_PATH_LEN
                        ),
                        max_visited=getattr(
                            self, "astar_max_visited", ASTAR_MAX_VISITED
                        ),
                        max_frontier=getattr(
                            self, "astar_max_frontier", ASTAR_MAX_FRONTIER
                        ),
                        wall_budget_s=getattr(
                            self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S
                        ),
                    )
                    check_ends = end_points
                    if path and self.env.is_valid_path(
                        check_ends, path, self.time, agv.name
                    ):
                        kin = self._append_kinematic_path(
                            agv,
                            path,
                            task,
                            loaded=False,
                            pickup_point=pk,
                            end_points=end_points,
                            src_steps=steps,
                            check_ends=check_ends,
                        )
                        if kin:
                            self.update_env(agv.name, kin)
                            fixed += 1
                            print(
                                f"[HEAL] continue assign {agv.name} {tid_s} "
                                f"@t={t_now} idle={idle}",
                                flush=True,
                            )
                            continue

            if idle < release_after:
                continue
            # Release ownership so heal_orphaned can restore surface head.
            print(
                f"[HEAL] release stale reservation {agv.name} {tid_s} "
                f"@t={t_now} idle={idle}",
                flush=True,
            )
            agv.task_id = None
            cool.pop(agv.name, None)
            assigned = getattr(self, "assigned_task_info", None)
            if isinstance(assigned, dict):
                assigned.pop(tid_s, None)
            inflight = getattr(self, "_inflight_tasks", None)
            if isinstance(inflight, dict):
                inflight.pop(agv.name, None)
            # Clear bogus pickup commit if cargo never left the station queue.
            committed = getattr(self, "_pickup_committed", None)
            if isinstance(committed, set):
                committed.discard(tid_s)
            self.tried_tasks = {
                (a, t) for (a, t) in self.tried_tasks if t != tid_s
            }
            fixed += 1
        if fixed:
            self.heal_orphaned_surface_tasks()
        return fixed

    def _pickup_name_for_task(self, task_id):
        for name, queue in self.task_states.items():
            if queue and queue[0].get('task_id') == task_id:
                return name
            # Promoted heads may leave task deeper in queue after assign reserve.
            for tinfo in queue or []:
                if isinstance(tinfo, dict) and tinfo.get("task_id") == task_id:
                    return name
        inflight = getattr(self, '_inflight_tasks', None) or {}
        for info in inflight.values():
            if info.get('task_id') == task_id:
                return info.get('pickup_name')
        assigned = getattr(self, "assigned_task_info", None) or {}
        info = assigned.get(str(task_id)) if isinstance(assigned, dict) else None
        if isinstance(info, dict) and info.get("pickup_name"):
            return info.get("pickup_name")
        return None

    def on_task_picked_up(self, task_id):
        """FIFO：物理取货（loaded 变 TRUE）后才 pop 队列并 promote 下一单。"""
        if task_id in self._pickup_committed:
            return
        pickup_name = self._pickup_name_for_task(task_id)
        if not pickup_name:
            return
        self._pickup_committed.add(task_id)
        print(f"FIFO 取货确认 {task_id} @ {pickup_name}")
        self.update_task_states(pickup_name)
        self.promote_station_after_pickup(pickup_name)

    def _step_at_time(self, agv, t):
        t = int(t)
        best = None
        for s in agv.steps:
            if int(s.get('timestamp', -1)) == t:
                return s
        for s in reversed(agv.steps):
            if int(s.get('timestamp', -1)) <= t:
                return s
        return best

    def _commit_pickups_at_time(self):
        t = int(self.time)
        for agv in self.agvs:
            tid = agv.task_id
            if not tid or str(tid).startswith('escape_'):
                continue
            if tid in self._pickup_committed:
                continue
            cur = self._step_at_time(agv, t)
            prev = self._step_at_time(agv, t - 1) if t > 0 else None
            if not cur:
                continue
            cur_loaded = str(cur.get('loaded', '')).lower() in ('true', '1', 'yes')
            prev_loaded = (
                str(prev.get('loaded', '')).lower() in ('true', '1', 'yes') if prev else False
            )
            if not cur_loaded or prev_loaded:
                continue
            step_tid = str(cur.get('task-id') or '')
            if step_tid and step_tid == str(tid):
                from ml_research.benchmarks.m5_replan import _at_task_pickup

                if not _at_task_pickup(self, agv, t):
                    continue
                self.on_task_picked_up(tid)

    def update_surface_tasks(self, name, task_id):
        """Legacy API: assign-time reserve only (promote deferred to pickup)."""
        self.reserve_assigned_task(task_id)
        
    def _pad_path_to_len(self, agv, target_len: int) -> None:
        """Ensure ``len(agv.path) == target_len`` with hold poses (index == time)."""
        if not agv.path:
            return
        last = agv.path[-1]
        while len(agv.path) < int(target_len):
            t = len(agv.path)
            agv.path.append((int(last[0]), int(last[1]), t, int(last[3]) % 360))
        self._sync_moving_obstacle_from_path(agv)

    def _sync_moving_obstacle_from_path(self, agv) -> None:
        """Occupancy table must match executed path (CSV comes from steps/path)."""
        if agv.path:
            self.env.moving_obstacles[agv.name] = list(agv.path)

    def _get_mo_horizon(self) -> int:
        """Active moving-obstacle reservation window (0 = legacy full freeze)."""
        h = getattr(self, "_mo_horizon", None)
        if h is not None:
            try:
                return max(0, int(h))
            except Exception:  # noqa: BLE001
                pass
        try:
            return max(0, int(os.environ.get("MOVING_OBSTACLE_HORIZON", "0") or "0"))
        except Exception:  # noqa: BLE001
            return max(0, int(MOVING_OBSTACLE_HORIZON or 0))

    def _hold_agv_at_tick(self, agv, t: int, xy, pitch: Optional[int] = None) -> None:
        """Force path[t] to stay at xy, drop future plan, sync steps/obstacles."""
        t = int(t)
        x, y = int(xy[0]), int(xy[1])
        if pitch is None:
            if t < len(agv.path):
                pitch = int(agv.path[t][3]) % 360
            elif agv.path:
                pitch = int(agv.path[-1][3]) % 360
            else:
                pitch = 90
        pitch = int(pitch) % 360
        pose = (x, y, t, pitch)
        if not agv.path:
            agv.path = [pose]
        elif t < len(agv.path):
            agv.path[t] = pose
            agv.path = list(agv.path[: t + 1])
        else:
            self._pad_path_to_len(agv, t)
            if t < len(agv.path):
                agv.path[t] = pose
            else:
                agv.path.append(pose)
            agv.path = list(agv.path[: t + 1])
        agv.state = pose
        agv.steps = [
            s for s in (agv.steps or []) if int(s.get("timestamp", -1)) <= t
        ]
        self._sync_step_pose_at(agv, t, x, y, pitch)
        # If hold left a rising-edge pickup off the pad, delay loaded metadata
        # (same idea as execution_shield — avoid pickup_cell validate fails).
        cur_s = None
        prev_s = None
        for s in agv.steps or []:
            ts = int(s.get("timestamp", -1))
            if ts == t:
                cur_s = s
            elif ts == t - 1:
                prev_s = s
        if cur_s is not None:
            loaded_cur = str(cur_s.get("loaded", "")).lower() in ("true", "1", "yes")
            loaded_prev = bool(
                prev_s is not None
                and str(prev_s.get("loaded", "")).lower() in ("true", "1", "yes")
            )
            if loaded_cur and not loaded_prev:
                pk = None
                tid = str(cur_s.get("task-id") or getattr(agv, "task_id", "") or "")
                info = self._task_info_for_agv(agv) if hasattr(self, "_task_info_for_agv") else None
                if isinstance(info, dict) and info.get("pickup_point"):
                    pk = (
                        int(info["pickup_point"][0]),
                        int(info["pickup_point"][1]),
                    )
                if pk is not None and (x, y) != pk:
                    cur_s["loaded"] = "FALSE"
                    cur_s["task-id"] = ""
                    cur_s["destination"] = ""
                    cur_s["Emergency"] = "FALSE"
        traj = self.env.moving_obstacles.get(agv.name)
        if traj is not None:
            if t < len(traj):
                traj = list(traj[: t + 1])
                traj[t] = pose
            elif t == len(traj):
                traj = list(traj) + [pose]
            else:
                while len(traj) < t:
                    last = traj[-1] if traj else pose
                    traj.append(
                        (int(last[0]), int(last[1]), len(traj), int(last[3]) % 360)
                    )
                traj.append(pose)
            self.env.moving_obstacles[agv.name] = traj
        self._exec_hold_events = int(getattr(self, "_exec_hold_events", 0) or 0) + 1

    def _resolve_execution_conflicts_hold_only(self) -> int:
        """Vertex/swap repair: losers hold at t-1; never nudge/teleport. Returns holds."""
        t = int(self.time)
        if t < 0:
            return 0
        intent = {}
        prev = {}
        pitch = {}
        for agv in self.agvs:
            if not agv.path:
                continue
            if t < len(agv.path):
                p = agv.path[t]
                intent[agv.name] = (int(p[0]), int(p[1]))
                pitch[agv.name] = int(p[3]) % 360
            else:
                p = agv.path[-1]
                intent[agv.name] = (int(p[0]), int(p[1]))
                pitch[agv.name] = int(p[3]) % 360
            if t > 0 and (t - 1) < len(agv.path):
                q = agv.path[t - 1]
                prev[agv.name] = (int(q[0]), int(q[1]))
                if agv.name not in pitch:
                    pitch[agv.name] = int(q[3]) % 360
            else:
                prev[agv.name] = intent[agv.name]
        if len(intent) < 2:
            return 0

        holds = 0
        names = list(intent.keys())
        # Edge swaps: both hold at previous cells.
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pa, pb = intent[a], intent[b]
                pra, prb = prev[a], prev[b]
                if pra == pb and prb == pa and pra != prb:
                    agv_a = next(x for x in self.agvs if x.name == a)
                    agv_b = next(x for x in self.agvs if x.name == b)
                    self._hold_agv_at_tick(agv_a, t, pra, pitch.get(a))
                    self._hold_agv_at_tick(agv_b, t, prb, pitch.get(b))
                    intent[a], intent[b] = pra, prb
                    holds += 2

        owners = defaultdict(list)
        for n, cell in intent.items():
            owners[cell].append(n)
        for cell, group in list(owners.items()):
            if len(group) <= 1:
                continue
            keep = None
            # Prefer agent whose pickup pad is this cell (rising-edge safe).
            for n in group:
                agv = next(x for x in self.agvs if x.name == n)
                info = self._task_info_for_agv(agv)
                pk = None
                if isinstance(info, dict) and info.get("pickup_point"):
                    pk = (
                        int(info["pickup_point"][0]),
                        int(info["pickup_point"][1]),
                    )
                if pk is not None and pk == cell:
                    keep = n
                    break
            if keep is None:
                for n in group:
                    if prev.get(n) == cell:
                        keep = n
                        break
            if keep is None:

                def _busy_key(n):
                    agv = next(x for x in self.agvs if x.name == n)
                    tid = getattr(agv, "task_id", None)
                    has_real = bool(tid) and not str(tid).startswith(
                        ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
                    )
                    return (0 if has_real else 1, n)

                keep = sorted(group, key=_busy_key)[0]
            for n in group:
                if n == keep:
                    continue
                agv = next(x for x in self.agvs if x.name == n)
                self._hold_agv_at_tick(agv, t, prev[n], pitch.get(n))
                intent[n] = prev[n]
                holds += 1
        return holds

    def _task_info_for_agv(self, agv) -> Optional[dict]:
        """Inflight / assigned payload for horizon continuation replans."""
        inflight = getattr(self, "_inflight_tasks", None) or {}
        info = inflight.get(agv.name)
        if isinstance(info, dict) and info.get("task_id"):
            return dict(info)
        tid = getattr(agv, "task_id", None)
        if not tid:
            return None
        tid_s = str(tid)
        assigned = getattr(self, "assigned_task_info", None) or {}
        if tid_s in assigned and isinstance(assigned[tid_s], dict):
            out = dict(assigned[tid_s])
            out.setdefault("task_id", tid_s)
            out.setdefault("agv", agv.name)
            return out
        surf = getattr(self, "surface_tasks", None) or {}
        if tid_s in surf and isinstance(surf[tid_s], dict):
            out = dict(surf[tid_s])
            out["task_id"] = tid_s
            out["agv"] = agv.name
            return out
        # Reserved tasks leave surface_tasks; recover from any station queue.
        for name, queue in (getattr(self, "task_states", None) or {}).items():
            for tinfo in queue or []:
                if not isinstance(tinfo, dict):
                    continue
                if str(tinfo.get("task_id") or "") != tid_s:
                    continue
                out = {k: v for k, v in tinfo.items()}
                if "end_points" in out:
                    out["end_points"] = list(out.get("end_points") or [])
                out["task_id"] = tid_s
                out["agv"] = agv.name
                out.setdefault("pickup_name", name)
                return out
        return None

    def _lookup_task_info_by_id(self, tid: str) -> Optional[dict]:
        """Find task payload by id across assigned / surface / queues."""
        tid_s = str(tid or "")
        if not tid_s:
            return None
        assigned = getattr(self, "assigned_task_info", None) or {}
        if tid_s in assigned and isinstance(assigned[tid_s], dict):
            out = dict(assigned[tid_s])
            out.setdefault("task_id", tid_s)
            return out
        surf = getattr(self, "surface_tasks", None) or {}
        if tid_s in surf and isinstance(surf[tid_s], dict):
            out = dict(surf[tid_s])
            out["task_id"] = tid_s
            return out
        for name, queue in (getattr(self, "task_states", None) or {}).items():
            for tinfo in queue or []:
                if not isinstance(tinfo, dict):
                    continue
                if str(tinfo.get("task_id") or "") != tid_s:
                    continue
                out = {k: v for k, v in tinfo.items()}
                if "end_points" in out:
                    out["end_points"] = list(out.get("end_points") or [])
                out["task_id"] = tid_s
                out.setdefault("pickup_name", name)
                return out
        return None

    def _replan_horizon_continuations(self) -> int:
        """When a truncated commit is nearly exhausted, plan the next horizon window."""
        h = self._get_mo_horizon()
        if h <= 0:
            return 0
        n_ok = 0
        t_now = int(self.time)
        cool = getattr(self, "_horizon_replan_at", None)
        if not isinstance(cool, dict):
            cool = {}
            self._horizon_replan_at = cool
        for agv in self.agvs:
            tid = getattr(agv, "task_id", None)
            if tid is None or str(tid).startswith(
                ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
            ):
                continue
            if not agv.path:
                continue
            # Still have plenty of committed horizon — skip.
            if len(agv.path) - 1 > t_now + 2:
                continue
            next_ok = int(cool.get(agv.name, 0) or 0)
            if t_now < next_ok:
                continue
            cool[agv.name] = t_now + 6
            if self._has_completed_unload_for_task(agv, str(tid)):
                continue
            info = self._task_info_for_agv(agv)
            if not info:
                continue
            pickup = info.get("pickup_point")
            end_points = list(info.get("end_points") or [])
            if pickup is None or len(pickup) < 2:
                continue
            if not end_points:
                continue
            start = agv.path[min(t_now, len(agv.path) - 1)]
            start_xy = (int(start[0]), int(start[1]), int(start[3]) % 360)
            step = self._step_at_time(agv, t_now)
            loaded = bool(
                step is not None
                and str(step.get("loaded", "")).lower() in ("true", "1", "yes")
            )
            # Already at goal pad while loaded → let unload logic finish.
            if loaded and (int(start[0]), int(start[1])) in {
                (int(e[0]), int(e[1])) for e in end_points if e is not None and len(e) >= 2
            }:
                continue
            task = {
                "agv": agv.name,
                "task_id": str(info.get("task_id") or tid),
                "destination": info.get("destination", ""),
                "priority": info.get("priority", "Normal"),
                "end_points": end_points,
                "force_full_path": True,
            }
            # Loaded: plan from current → dropoff. Else continue toward pickup.
            if loaded:
                pk = (int(start[0]), int(start[1]))
            else:
                pk = (int(pickup[0]), int(pickup[1]))
            path, steps = A_Star(
                task,
                start_xy,
                pk,
                end_points,
                self.env.get_static_obstacles(),
                self.env.moving_obstacles,
                max_path_len=getattr(self, "astar_max_path_len", ASTAR_MAX_PATH_LEN),
                max_visited=getattr(self, "astar_max_visited", ASTAR_MAX_VISITED),
                max_frontier=getattr(self, "astar_max_frontier", ASTAR_MAX_FRONTIER),
                wall_budget_s=getattr(self, "astar_wall_budget_s", ASTAR_WALL_BUDGET_S),
            )
            check_ends = end_points if loaded else [pk]
            if not path or not self.env.is_valid_path(
                check_ends, path, self.time, agv.name
            ):
                continue
            kin = self._append_kinematic_path(
                agv,
                path,
                task,
                loaded=loaded,
                pickup_point=pk if not loaded else None,
                end_points=end_points,
                src_steps=steps,
                check_ends=check_ends,
            )
            if kin:
                self.update_env(agv.name, kin)
                n_ok += 1
                self._horizon_replan_count = (
                    int(getattr(self, "_horizon_replan_count", 0) or 0) + 1
                )
        return n_ok

    def _compute_kinematic_append(
        self,
        agv,
        path,
        task: dict,
        *,
        loaded: bool = False,
        pickup_point=None,
        end_points=None,
        src_steps=None,
    ):
        """Build the kinematic segment that would be appended (no mutation).

        Returns ``(kin, step_rows)`` or ``(None, None)``.
        """
        if not path:
            return None, None
        cur = agv.path[-1] if agv.path else agv.state
        if cur is None:
            cur = tuple(path[0])
        path_l = [tuple(p) for p in path]
        # A* already includes start. Prepending it again + wait-preserve would
        # insert a duplicate hold and shift every later occupancy tick.
        if (
            path_l
            and tuple(path_l[0][:2]) == tuple(cur[:2])
            and int(path_l[0][3]) % 360 == int(cur[3]) % 360
        ):
            seeded = path_l
        else:
            seeded = [tuple(cur)] + path_l
        kin = expand_path_turn_moves(seeded)
        if len(kin) <= 1:
            return None, None
        kin = kin[1:]
        base_t = len(agv.path)
        if agv.path:
            base_t = max(base_t, int(agv.path[-1][2]) + 1)
        else:
            base_t = max(0, int(cur[2]) + 1)
        # Attach at next free tick; pad is applied only on commit so preview
        # timestamps match the eventual committed index==time layout.
        attach_t = len(agv.path) if agv.path else base_t
        if agv.path:
            attach_t = max(attach_t, int(agv.path[-1][2]) + 1)
        kin = retime_path_from(kin, attach_t)
        # Competition: pickup / unload each need 1s in-place dwell.
        ends = list(end_points or (task.get("end_points") if task else None) or [])
        pk_xy = None
        if pickup_point is not None and len(pickup_point) >= 2:
            pk_xy = (int(pickup_point[0]), int(pickup_point[1]))
        cur_xy = (int(cur[0]), int(cur[1]))
        # Path ended on the pad then delivery replan drops the start cell
        # (kin[1:]). Without a same-cell dwell here, loaded rises on the exit
        # move → pickup_dwell INVALID (e.g. Dragon pad (2,10)→(2,11)).
        if (
            pk_xy is not None
            and cur_xy == pk_xy
            and kin
            and (int(kin[0][0]), int(kin[0][1])) != pk_xy
        ):
            dwell = (pk_xy[0], pk_xy[1], int(attach_t), int(cur[3]) % 360)
            kin = retime_path_from([dwell] + [tuple(p) for p in kin], attach_t)
        kin = inject_pickup_unload_dwells(
            kin, pickup_point=pickup_point, end_points=ends
        )
        step_rows = steps_for_kinematic_path(
            kin,
            task,
            loaded=loaded,
            pickup_point=pickup_point,
            end_points=end_points,
            src_steps=src_steps,
        )
        return kin, step_rows

    def _append_kinematic_path(
        self,
        agv,
        path,
        task: dict,
        *,
        loaded: bool = False,
        pickup_point=None,
        end_points=None,
        src_steps=None,
        check_ends=None,
    ):
        """Append kinematic path; optionally reject if spacetime-invalid after expand.

        A* validates the raw search path, but ``expand_path_turn_moves`` inserts
        in-place turns and ``retime_path_from`` shifts occupancy in time. The
        committed trajectory must be re-checked — otherwise CSV collisions appear
        despite a "valid" A* result.

        Returns the kinematic segment appended (for ``update_env``), or ``None``.
        """
        kin, step_rows = self._compute_kinematic_append(
            agv,
            path,
            task,
            loaded=loaded,
            pickup_point=pickup_point,
            end_points=end_points,
            src_steps=src_steps,
        )
        if not kin:
            return None
        ends = check_ends
        if ends is None:
            ends = list(end_points or task.get("end_points") or [])
        # Pad holds are occupancy too — check them with the expanded segment.
        pad_poses = []
        if agv.path:
            attach_t = int(kin[0][2])
            last = agv.path[-1]
            t = len(agv.path)
            while t < attach_t:
                pad_poses.append(
                    (int(last[0]), int(last[1]), t, int(last[3]) % 360)
                )
                t += 1
        check_path = pad_poses + list(kin)
        if path_has_spacetime_conflict(
            check_path, self.env.moving_obstacles, getattr(agv, "name", None)
        ):
            print(
                f"[A*] reject after kinematic expand AGV {getattr(agv, 'name', '?')} "
                f"task {task.get('task_id')} len={len(kin)} @t={self.time}",
                flush=True,
            )
            return None
        if ends:
            end_xy = {
                (int(e[0]), int(e[1]))
                for e in ends
                if e is not None and len(e) >= 2
            }
            if end_xy and not any(tuple(p[:2]) in end_xy for p in kin):
                return None
        if agv.path:
            attach_t = int(kin[0][2])
            self._pad_path_to_len(agv, attach_t)
        agv.path += kin
        agv.steps += step_rows
        self._sync_moving_obstacle_from_path(agv)
        return kin

    def update_agvs(self, assigned_task, path, steps, check_ends=None):
        """Append kinematic path; return segment for ``update_env`` (or None)."""
        for agv in self.agvs:
            if agv.name != assigned_task['agv']:
                continue
            new_tid = str(assigned_task.get("task_id") or "")
            # Hard guard: never rebind a different real task while cargo is loaded.
            # (Caused Horse-6→Ox-3 mid-carry: no FALSE→TRUE → FIFO/display fail.)
            step = self._step_at_time(agv, int(self.time))
            if step is not None:
                loaded = str(step.get("loaded", "")).lower() in ("true", "1", "yes")
                cur_tid = str(step.get("task-id") or getattr(agv, "task_id", "") or "").strip()
                if (
                    loaded
                    and cur_tid
                    and new_tid
                    and cur_tid != new_tid
                    and not cur_tid.startswith(
                        ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
                    )
                    and not new_tid.startswith(
                        ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
                    )
                ):
                    print(
                        f"[GUARD] refuse assign {new_tid} over loaded {cur_tid} "
                        f"on {agv.name} @t={self.time}",
                        flush=True,
                    )
                    return None
            task = {
                "agv": assigned_task["agv"],
                "task_id": assigned_task.get("task_id", ""),
                "destination": assigned_task.get("destination", ""),
                "priority": assigned_task.get("priority", "Normal"),
                "end_points": list(assigned_task.get("end_points") or []),
            }
            kin = self._append_kinematic_path(
                agv,
                path,
                task,
                loaded=False,
                pickup_point=assigned_task.get("pickup_point"),
                end_points=assigned_task.get("end_points"),
                src_steps=steps,
                check_ends=check_ends,
            )
            if kin is None:
                return None
            agv.task_id = assigned_task['task_id']
            agv.priority = assigned_task['priority']
            # Keep payload for path-end continue/heal after surface reserve.
            store = getattr(self, "assigned_task_info", None)
            if store is None:
                store = {}
                self.assigned_task_info = store
            if isinstance(store, dict):
                tid_s = str(assigned_task.get("task_id") or "")
                if tid_s:
                    store[tid_s] = {
                        "task_id": tid_s,
                        "agv": assigned_task.get("agv"),
                        "destination": assigned_task.get("destination", ""),
                        "priority": assigned_task.get("priority", "Normal"),
                        "end_points": list(assigned_task.get("end_points") or []),
                        "pickup_point": assigned_task.get("pickup_point"),
                        "pickup_name": assigned_task.get("pickup_name")
                        or self._pickup_name_for_task(tid_s),
                    }
            return kin
        return None
                    
    def update_env(self, agv_name, path):
        if agv_name not in self.env.moving_obstacles:
            self.env.moving_obstacles[agv_name] = []
        existing = self.env.moving_obstacles[agv_name]
        if existing and path:
            last_t = existing[-1][2]
            path = [p for p in path if p[2] > last_t]
        self.env.moving_obstacles[agv_name] += path
            
    def update_tried_tasks(self, agv_name):
        # 当任务成功完成时，从 tried_tasks 中移除该 AGV 的所有失败记录
        # 因为该 AGV 现在可以接受新任务了
        self.tried_tasks = {(agv, task) for agv, task in self.tried_tasks if agv != agv_name}
        self.assign_cooldown.pop(agv_name, None)
        print(f"AGV {agv_name} 任务找到路径，当前 tried_tasks: {self.tried_tasks}")
            
            
    def _prev_pos_at(self, agv_name, t):
        """其他车在时刻 t-1 的位置（用于避免被动挪车造成对穿）。"""
        traj = self.env.moving_obstacles.get(agv_name)
        if not traj:
            return None
        if t - 1 < len(traj):
            return traj[t - 1][:2]
        return traj[-1][:2]

    def _sync_step_pose_at(self, agv, t: int, x: int, y: int, pitch: int) -> None:
        """Keep steps / moving_obstacles aligned when path[t] is rewritten."""
        t = int(t)
        x, y, pitch = int(x), int(y), int(pitch) % 360
        found = False
        for s in agv.steps:
            if int(s.get("timestamp", -1)) == t:
                s["X"], s["Y"], s["pitch"] = x, y, pitch
                found = True
                break
        if not found and agv.steps:
            # Carry metadata from nearest prior step.
            prev = None
            for s in agv.steps:
                if int(s.get("timestamp", -1)) <= t:
                    prev = s
            base = prev or agv.steps[-1]
            agv.steps.append(
                {
                    "timestamp": t,
                    "name": agv.name,
                    "X": x,
                    "Y": y,
                    "pitch": pitch,
                    "loaded": base.get("loaded", "FALSE"),
                    "destination": base.get("destination", ""),
                    "Emergency": base.get("Emergency", "FALSE"),
                    "task-id": base.get("task-id", ""),
                }
            )
        traj = self.env.moving_obstacles.get(agv.name)
        if traj is not None:
            pose = (x, y, t, pitch)
            if t < len(traj):
                traj[t] = pose
            elif t == len(traj):
                traj.append(pose)
            else:
                while len(traj) < t:
                    last = traj[-1] if traj else pose
                    traj.append(
                        (int(last[0]), int(last[1]), len(traj), int(last[3]) % 360)
                    )
                traj.append(pose)
            self.env.moving_obstacles[agv.name] = traj

    def _cell_reserved_at(self, t: int, xy, skip_name=None) -> bool:
        """True if another AGV's committed path already occupies xy at tick t."""
        t = int(t)
        cell = (int(xy[0]), int(xy[1]))
        for other in self.agvs:
            if skip_name and other.name == skip_name:
                continue
            if other.path and t < len(other.path) and tuple(other.path[t][:2]) == cell:
                return True
        return False

    def _enforce_unit_step_at(self, agv, t: int) -> bool:
        """Root kinematic invariant: path[t] may only hold/turn or move one cell.

        Writers (assign append, conflict shield, relocate) can leave future
        waypoints on an obsolete plan after the current tick is displaced.
        Clamping here — every tick, before occupancy — removes teleports at the
        execution layer instead of chasing each writer.

        Never steps onto static / maze walls (SH03 Ironhide (5,6) regression).
        Never rewrite into a cell another AGV already reserved at t (that was
        the SH01 kin-reject leftover: leader hold+turn while follower enters).
        """
        t = int(t)
        if t <= 0 or not agv.path or t >= len(agv.path) or (t - 1) >= len(agv.path):
            return False
        px, py = int(agv.path[t - 1][0]), int(agv.path[t - 1][1])
        pp = int(agv.path[t - 1][3]) % 360
        cx, cy = int(agv.path[t][0]), int(agv.path[t][1])
        cp = int(agv.path[t][3]) % 360
        manh = abs(cx - px) + abs(cy - py)
        pitch_of = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270}
        static = set()
        try:
            static = {
                (int(p[0]), int(p[1]))
                for p in (self.env.get_static_obstacles() or [])
            }
        except Exception:
            static = set()
        hw = getattr(self.env, "_extra_obstacle_cells", None) or set()
        blocked = set(static) | set(hw)

        def _safe(nx: int, ny: int) -> bool:
            if not (1 <= nx <= 20 and 1 <= ny <= 20):
                return False
            if (nx, ny) in blocked:
                return False
            if self._cell_reserved_at(t, (nx, ny), skip_name=agv.name):
                return False
            return True

        def _write(nx: int, ny: int, pitch: int) -> bool:
            if self._cell_reserved_at(t, (nx, ny), skip_name=agv.name):
                return False
            agv.path[t] = (int(nx), int(ny), t, int(pitch) % 360)
            self._sync_step_pose_at(agv, t, nx, ny, pitch)
            return True

        if manh == 0:
            # Sitting on a wall: nudge to previous free cell.
            if (cx, cy) in blocked and _safe(px, py):
                return _write(px, py, pp)
            return False
        if manh == 1:
            if (cx, cy) in blocked:
                # Planned into a wall — hold previous cell (turn toward goal).
                dx = 0 if cx == px else (1 if cx > px else -1)
                dy = 0 if cy == py else (1 if cy > py else -1)
                need = pitch_of.get((dx, dy), pp)
                if _safe(px, py):
                    return _write(px, py, need)
                return False
            need = pitch_of[(cx - px, cy - py)]
            if cp == need:
                return False
            # Competition forbids move+turn. Prefer strafe: keep previous pitch.
            return _write(cx, cy, pp)

        # manh >= 2: one legal action toward the planned cell (turn then move).
        dx = 0 if cx == px else (1 if cx > px else -1)
        dy = 0 if cy == py else (1 if cy > py else -1)
        # Prefer axis that reduces error; x-first matches expand_path_turn_moves.
        if dx != 0:
            need = pitch_of[(dx, 0)]
            if pp != need:
                if _safe(px, py):
                    return _write(px, py, need)
                return False
            nx, ny = px + dx, py
            if not _safe(nx, ny):
                # Try the other axis, else hold+turn if prev is free.
                if dy != 0 and _safe(px, py + dy):
                    need2 = pitch_of[(0, dy)]
                    if pp != need2:
                        return _write(px, py, need2)
                    return _write(px, py + dy, need2)
                if _safe(px, py):
                    return _write(px, py, need)
                return False
            return _write(nx, ny, need)
        need = pitch_of[(0, dy)]
        if pp != need:
            if _safe(px, py):
                return _write(px, py, need)
            return False
        nx, ny = px, py + dy
        if not _safe(nx, ny):
            if _safe(px, py):
                return _write(px, py, need)
            return False
        return _write(nx, ny, need)

    def update_agv_state(self):
        # Clamp before occupancy: shield/replan may have left path[t] discontinuous
        # with the *executed* path[t-1].
        if self.time > 0:
            for agv in self.agvs:
                self._enforce_unit_step_at(agv, self.time)

        def _carry_meta(agv):
            """Preserve loaded/task fields when extending past path end."""
            last_s = agv.steps[-1] if agv.steps else {}
            loaded = str(last_s.get("loaded", "")).lower() in ("true", "1", "yes")
            # Also trust agv.task_id if steps were already cleared incorrectly.
            tid = str(last_s.get("task-id") or "").strip()
            if not tid and agv.task_id and not str(agv.task_id).startswith(
                ("escape_", "fsm_", "yield_", "stage_")
            ):
                tid = str(agv.task_id)
            # Only keep carrying if we truly picked up (FIFO commit) or last
            # step was loaded. Do NOT invent loaded=True from bare task_id —
            # that parked unfinished reservations forever after path end.
            committed = getattr(self, "_pickup_committed", None) or set()
            if tid and not self._has_completed_unload_for_task(agv, tid):
                if loaded or tid in committed:
                    loaded = True
                else:
                    loaded = False
            if not loaded:
                return {
                    "loaded": "FALSE",
                    "destination": "",
                    "Emergency": "FALSE",
                    "task-id": "",
                }
            return {
                "loaded": "TRUE",
                "destination": str(last_s.get("destination") or ""),
                "Emergency": str(last_s.get("Emergency") or "FALSE"),
                "task-id": tid,
            }

        # 记录本时刻已被占用的格子，以及“从哪格驶入”（防对穿）
        claimed = {}
        arrive_from = {}
        for agv in self.agvs:
            if agv.path and self.time < len(agv.path):
                pos = agv.path[self.time][:2]
                claimed[pos] = agv.name
                if self.time > 0 and self.time - 1 < len(agv.path):
                    arrive_from[agv.name] = agv.path[self.time - 1][:2]
                else:
                    arrive_from[agv.name] = self._prev_pos_at(agv.name, self.time)

        for agv in self.agvs:
            if self.time < len(agv.path):
                agv.state = agv.path[self.time]
            else:
                if not agv.path:
                    continue
                last = agv.path[-1]
                pos = last[:2]
                meta = _carry_meta(agv)
                # 若终点格已被其他车占用，尝试挪一格再等待（禁止与驶入车对穿）
                if pos in claimed and claimed[pos] != agv.name:
                    moved = False
                    static_obs = set(self.env.get_static_obstacles())
                    blocker = claimed[pos]
                    blocker_from = arrive_from.get(blocker)
                    cur_pitch = int(last[3]) % 360
                    # Prefer continuing current heading; otherwise turn in place this tick.
                    move_order = [(1, 0, 0), (-1, 0, 180), (0, 1, 90), (0, -1, 270)]
                    move_order.sort(key=lambda m: 0 if m[2] == cur_pitch else 1)
                    for dx, dy, pitch in move_order:
                        cand = (pos[0] + dx, pos[1] + dy)
                        if not _in_playable(cand[0], cand[1]):
                            continue
                        if cand in static_obs or cand in claimed:
                            continue
                        # 对穿：对方从 cand 驶入 pos，我们同时从 pos 驶入 cand
                        if blocker_from == cand:
                            continue
                        # 也不要驶入其他车本时刻离开后腾出、但会造成交换的格
                        swap_risk = False
                        for other in self.agvs:
                            if other.name == agv.name:
                                continue
                            if other.path and self.time < len(other.path):
                                if other.path[self.time][:2] == pos and \
                                   self.time > 0 and self.time - 1 < len(other.path) and \
                                   other.path[self.time - 1][:2] == cand:
                                    swap_risk = True
                                    break
                        if swap_risk:
                            continue
                        # Cell is claimed — cannot turn in place here (that was writing
                        # vertex collisions). Only leave if already facing the exit.
                        if pitch != cur_pitch:
                            continue
                        step = (cand[0], cand[1], self.time, pitch)
                        agv.path.append(step)
                        agv.steps.append({
                            "timestamp": self.time,
                            "name": agv.name,
                            "X": step[0],
                            "Y": step[1],
                            "pitch": step[3],
                            **meta,
                        })
                        if agv.name in self.env.moving_obstacles:
                            self.env.moving_obstacles[agv.name].append(step)
                        agv.state = step
                        claimed[step[:2]] = agv.name
                        moved = True
                        break
                    if not moved:
                        # Never record idle AGV on a cell another AGV already reserved.
                        occupied = set(claimed.keys()) | set(static_obs)
                        goal = self._pick_holding_cell(
                            pos, set(), occupied, static_obs, min_r=1, max_r=6
                        )
                        if goal is not None and self._relocate_idle_agv(
                            agv, goal, occupied
                        ):
                            if self.time < len(agv.path):
                                agv.state = agv.path[self.time]
                                claimed[tuple(agv.state[:2])] = agv.name
                                moved = True
                        if not moved:
                            # Last resort: step onto any free orthogonal neighbor.
                            for dx, dy, pitch in move_order:
                                cand = (pos[0] + dx, pos[1] + dy)
                                if not _in_playable(cand[0], cand[1]):
                                    continue
                                if cand in static_obs or cand in claimed:
                                    continue
                                step = (cand[0], cand[1], self.time, pitch)
                                agv.path.append(step)
                                agv.steps.append({
                                    "timestamp": self.time,
                                    "name": agv.name,
                                    "X": step[0],
                                    "Y": step[1],
                                    "pitch": step[3],
                                    **meta,
                                })
                                if agv.name in self.env.moving_obstacles:
                                    self.env.moving_obstacles[agv.name].append(step)
                                agv.state = step
                                claimed[step[:2]] = agv.name
                                moved = True
                                break
                        if not moved:
                            print(
                                f"[A*] idle refuse collide {agv.name} @ {pos} "
                                f"t={self.time} blocked_by={blocker}",
                                flush=True,
                            )
                else:
                    agv_state_temp = (*pos, self.time, last[3])
                    agv.state = agv_state_temp
                    agv.path.append(agv_state_temp)
                    agv.steps.append({
                        "timestamp": self.time,
                        "name": agv.name,
                        "X": pos[0],
                        "Y": pos[1],
                        "pitch": last[3],
                        **meta,
                    })
                    if agv.name in self.env.moving_obstacles:
                        self.env.moving_obstacles[agv.name].append(agv_state_temp)
                    claimed[pos] = agv.name

            # Only clear real tasks after unload is done — never on mid-trip /
            # parking / staged path end (that orphaned reserved surface tasks).
            if self.time >= len(agv.path) - 1 and agv.path:
                if self._should_clear_task_on_path_end(agv):
                    old_tid = str(getattr(agv, "task_id", "") or "")
                    agv.task_id = None
                    store = getattr(self, "assigned_task_info", None)
                    if old_tid and isinstance(store, dict):
                        store.pop(old_tid, None)

        self._commit_pickups_at_time()

    def _task_unload_pads(self, tid: str):
        """Leave-ring pads for this task's destination (not any hub)."""
        info = None
        tid_s = str(tid)
        for store in (
            getattr(self, "surface_tasks", None),
            getattr(self, "assigned_task_info", None),
            getattr(self, "_inflight_tasks", None),
        ):
            if not store:
                continue
            if tid_s in store:
                info = store[tid_s]
                break
            if isinstance(store, dict):
                for v in store.values():
                    if isinstance(v, dict) and str(v.get("task_id") or "") == tid_s:
                        info = v
                        break
            if info is not None:
                break
        # Fall back: destination name from the AGV still carrying this task.
        dest_name = ""
        if isinstance(info, dict):
            dest_name = str(info.get("destination") or "").strip()
        if not dest_name:
            for agv in getattr(self, "agvs", []) or []:
                if str(getattr(agv, "task_id", "") or "") != tid_s:
                    continue
                for s in reversed(getattr(agv, "steps", None) or []):
                    if str(s.get("task-id") or "") == tid_s and s.get("destination"):
                        dest_name = str(s.get("destination") or "").strip()
                        break
                if dest_name:
                    break
        pads = set()
        if isinstance(info, dict):
            for ep in info.get("end_points") or ():
                if ep is not None and len(ep) >= 2:
                    pads.add((int(ep[0]), int(ep[1])))
        ends = getattr(self.env, "destination_points", None)
        if dest_name and isinstance(ends, dict) and dest_name in ends:
            cx, cy = int(ends[dest_name][0]), int(ends[dest_name][1])
            pads.add((cx, cy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pads.add((cx + dx, cy + dy))
        # Recover pads from remaining/queued task_states by destination name.
        if dest_name and not pads:
            for queue in (getattr(self, "task_states", None) or {}).values():
                for tinfo in queue or []:
                    if not isinstance(tinfo, dict):
                        continue
                    if str(tinfo.get("destination") or "").strip() != dest_name:
                        continue
                    for ep in tinfo.get("end_points") or ():
                        if ep is not None and len(ep) >= 2:
                            pads.add((int(ep[0]), int(ep[1])))
                    if pads:
                        break
                if pads:
                    break
        if dest_name and not pads and hasattr(self.env, "get_end_points"):
            try:
                for ep in self.env.get_end_points(dest_name) or []:
                    pads.add((int(ep[0]), int(ep[1])))
            except Exception:  # noqa: BLE001
                pass
        return pads

    def _is_unload_interaction_cell(self, pos, tid=None) -> bool:
        """True if pos is a leave-ring slot of the task destination (or any hub)."""
        x, y = int(pos[0]), int(pos[1])
        if tid:
            pads = self._task_unload_pads(str(tid))
            # Task-scoped: never fall back to "any hub" (Shanghai≠Beijing).
            return bool(pads) and (x, y) in pads
        for dest in self.env.destination_points:
            dx, dy = int(dest[0]), int(dest[1])
            if abs(x - dx) + abs(y - dy) == 1:
                return True
        return False

    def _has_completed_unload_for_task(self, agv, tid: str) -> bool:
        """TRUE→FALSE unload on this task's pad for the *latest* loaded streak.

        Must not treat an earlier finished run of the same ``tid`` as completion
        while a later re-carry is still loaded (that cleared ``task_id`` mid-trip
        and allowed a different task to overwrite cargo metadata).
        """
        tid_s = str(tid)
        pads = self._task_unload_pads(tid_s)
        steps = list(agv.steps or [])
        last_load_i = None
        for i, s in enumerate(steps):
            stid = str(s.get("task-id") or "")
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            if stid == tid_s and loaded:
                last_load_i = i
        if last_load_i is None:
            return False
        # Still carrying this tid at/after the latest load index with no unload.
        for j in range(last_load_i + 1, len(steps)):
            s = steps[j]
            stid = str(s.get("task-id") or "")
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            if stid == tid_s and loaded:
                continue
            if loaded and stid and stid != tid_s:
                # Mid-carry rebind — not a completed unload.
                return False
            if not loaded:
                try:
                    pos = (int(s["X"]), int(s["Y"]))
                except Exception:
                    return False
                if pads:
                    return pos in pads
                return self._is_unload_interaction_cell(pos, tid=tid_s)
            return False
        return False

    def _should_clear_task_on_path_end(self, agv) -> bool:
        """Path end clears task_id only for pseudo-tasks or finished unload."""
        tid = agv.task_id
        if tid is None:
            return False
        s = str(tid)
        # Ephemeral relocate / park / yield labels — OK to clear when path ends
        if s.startswith(("escape_", "fsm_", "yield_", "stage_")):
            return True
        # Real MAPD task: keep until unload completed at a dropoff slot
        return self._has_completed_unload_for_task(agv, s)
                
            
    def all_over(self):
        for agv in self.agvs:
            if agv.task_id is not None:
                return False
        if self.surface_tasks or self.task_states:
            return False
        return True
    
    def steps_reorganize(self):
        steps_dict = {}
        for agv in self.agvs:
            for step in agv.steps:
                if step['timestamp'] not in steps_dict:
                    steps_dict[step["timestamp"]] = []
                steps_dict[step['timestamp']].append(step)
        return steps_dict

                 
#  assigned_tasks.append({
#                 "agv": agv_id_min,
#                 "task": task_id_min,
#                 "agv_start_point": agv_states[agv_id_min]['pos'],
#                 "pickup_point": task_states[task_name_min][0]['pickup_point'],
#                 "end_points": task_states[task_name_min][0]['end_points'],
#                 "destination": task_states[task_name_min][0]['destination'],
#                 "priority": task_states[task_name_min][0]['priority'],
#             })   

class ENV:
    def __init__(self, start_points, destination_points):
        self.start_points = start_points
        self.destination_points = destination_points
        self.moving_obstacles = {}
    
    def get_end_points(self, destination_name):
        possible_pos = [
        (1, 0),    # 右
        (-1, 0), # 左
        (0, 1),   # 上
        (0, -1)  # 下
        ]
        # print(end_point)
        # print(end_point_name)
        destination = self.destination_points[destination_name]
        possible_end_points = []
        for x,y in possible_pos:
            possible_end_points.append((x+destination[0],y+destination[1]))
        # print(possible_end_point)
        return possible_end_points
    

        
    '''
    用于Astar检查
    '''
    def get_static_obstacles(self):
        return self.start_points + self.destination_points
        

    
    '''
    用于simulation验证
    '''
    def is_valid_path(self, end_points, new_path, time, agv_name=None):
        # 路径中必须完成卸货（访问过卸货点）；允许随后驶离让位
        if not new_path:
            return False
        
        if not any(p[:2] in end_points for p in new_path):
            return False

        if path_has_spacetime_conflict(new_path, self.moving_obstacles, agv_name):
            return False
        return True

    def path_has_conflict(self, new_path, agv_name=None) -> bool:
        return path_has_spacetime_conflict(new_path, self.moving_obstacles, agv_name)
                    
        
        
    
    
class AGV:
    def __init__(self, name, state, task_id, priority=False, path=None, steps=None):
        self.name = name
        self.state = state
        self.task_id = task_id
        self.priority = priority
        self.path = path if path is not None else []
        self.steps = steps if steps is not None else []
    
    def __repr__(self):
        return 'AGV:' + str(self.name) + '= '+ ', state:' + str(self.state) + ', task_id:' +\
        str(self.task_id) + ', priority:' + str(self.priority) + ', path:' + str(self.path) + '\n'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def _pitch_turn_sequence(d_from, d_to):
    """Intermediate headings (±90°/tick) from d_from to d_to, excluding d_from."""
    d_from = int(d_from) % 360
    d_to = int(d_to) % 360
    if d_from == d_to:
        return []
    cw = (d_to - d_from) % 360
    if cw <= 180:
        n = max(1, cw // 90)
        return [(d_from + 90 * i) % 360 for i in range(1, n + 1)]
    n = max(1, (360 - cw) // 90)
    return [(d_from - 90 * i) % 360 for i in range(1, n + 1)]


def inject_pickup_unload_dwells(kin, pickup_point=None, end_points=None):
    """Ensure 1s in-place wait at pickup / unload so loaded flips can dwell.

    Competition: pickup and unload each need 1s on the same cell (cannot flip
    ``loaded`` while translating). Inserts missing stay poses then retimes.
    """
    if not kin:
        return []
    poses = [tuple(p) for p in kin]
    pickup_xy = None
    if pickup_point is not None and len(pickup_point) >= 2:
        pickup_xy = (int(pickup_point[0]), int(pickup_point[1]))
    end_set = set()
    for ep in end_points or ():
        if ep is not None and len(ep) >= 2:
            end_set.add((int(ep[0]), int(ep[1])))

    if pickup_xy is not None:
        for i, p in enumerate(poses):
            if (int(p[0]), int(p[1])) != pickup_xy:
                continue
            nxt = poses[i + 1] if i + 1 < len(poses) else None
            if nxt is None or (int(nxt[0]), int(nxt[1])) != pickup_xy:
                poses.insert(
                    i + 1,
                    (int(p[0]), int(p[1]), int(p[2]) + 1, int(p[3]) % 360),
                )
            break

    if end_set:
        pick_i = -1
        if pickup_xy is not None:
            for i, p in enumerate(poses):
                if (int(p[0]), int(p[1])) == pickup_xy:
                    pick_i = i
                    break
        for i in range(pick_i + 1, len(poses)):
            cell = (int(poses[i][0]), int(poses[i][1]))
            if cell not in end_set:
                continue
            # Need ≥2 consecutive ticks on pad (arrive + unload dwell).
            j = i
            while (
                j + 1 < len(poses)
                and (int(poses[j + 1][0]), int(poses[j + 1][1])) == cell
            ):
                j += 1
            if j == i:
                p = poses[i]
                poses.insert(
                    i + 1,
                    (int(p[0]), int(p[1]), int(p[2]) + 1, int(p[3]) % 360),
                )
            break

    t0 = int(poses[0][2]) if poses else 0
    return retime_path_from(poses, t0)


def expand_path_turn_moves(path):
    """Kinematic sanitize: unit steps only; in-place turn before any heading change.

    Guarantees consecutive poses are either:
      - same cell with pitch change of ±90° (turn; 180° uses two ticks), or
      - adjacent cell with pitch matching the move direction.
    Multi-cell gaps are Manhattan-interpolated (no teleports).
    """
    if not path:
        return []
    pitch_of = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270}
    out = [tuple(path[0])]
    t = int(out[0][2])

    def emit(x, y, pitch):
        nonlocal t
        t += 1
        out.append((int(x), int(y), int(t), int(pitch) % 360))

    def emit_turns(x, y, from_p, to_p):
        nonlocal t
        cur = int(from_p) % 360
        for h in _pitch_turn_sequence(cur, to_p):
            emit(x, y, h)
            cur = h
        return cur

    waypoints = [tuple(p) for p in path[1:]]
    for idx, cur in enumerate(waypoints):
        px, py, _, pp = out[-1]
        pp = int(pp) % 360
        cx, cy = int(cur[0]), int(cur[1])
        want = int(cur[3]) % 360 if len(cur) > 3 else pp
        is_last = idx == len(waypoints) - 1

        x, y = int(px), int(py)
        # Stay-in-place: keep A* wait ticks (same heading) and in-place turns.
        # Dropping waits packed later cells onto other AGVs' reserved times.
        if (x, y) == (cx, cy):
            if pp != want:
                emit_turns(x, y, pp, want)
            else:
                emit(x, y, pp)
            continue

        # Walk Manhattan: one axis at a time (x then y)
        while x != cx:
            dx = 1 if cx > x else -1
            need = pitch_of[(dx, 0)]
            if pp != need:
                pp = emit_turns(x, y, pp, need)
            x += dx
            emit(x, y, pp)
        while y != cy:
            dy = 1 if cy > y else -1
            need = pitch_of[(0, dy)]
            if pp != need:
                pp = emit_turns(x, y, pp, need)
            y += dy
            emit(x, y, pp)

        # Only force declared facing on the final waypoint (avoid spurious re-turns).
        if is_last and pp != want:
            emit_turns(x, y, pp, want)

    return out


def retime_path_from(path, t0: int):
    """Force path timestamps to t0, t0+1, ... (list index == time)."""
    t0 = int(t0)
    return [
        (int(p[0]), int(p[1]), t0 + i, int(p[3]) % 360)
        for i, p in enumerate(path)
    ]


def steps_for_kinematic_path(
    kin,
    task,
    *,
    loaded: bool = False,
    pickup_point=None,
    end_points=None,
    src_steps=None,
):
    """Build step rows for an already-expanded kinematic path.

    Full assign paths must be:
      unloaded → (reach pickup) → loaded → (unload pad) → unloaded leave
    A single ``path_to_step(..., loaded=True)`` on start→leave wrongly puts the
    only FALSE tick on a non-pad cell; the conflict shield then restores TRUE
    and the AGV never clears the task.
    """
    if not kin:
        return []
    has_loaded = bool(loaded)
    if src_steps:
        has_loaded = has_loaded or any(
            str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            for s in src_steps
        )
    pickup_xy = None
    if pickup_point is not None and len(pickup_point) >= 2:
        pickup_xy = (int(pickup_point[0]), int(pickup_point[1]))
    end_set = set()
    for ep in end_points or task.get("end_points") or ():
        if ep is not None and len(ep) >= 2:
            end_set.add((int(ep[0]), int(ep[1])))

    if not has_loaded:
        return path_to_step(kin, task, False, expand=False)

    if pickup_xy is None and not end_set:
        return path_to_step(kin, task, True, expand=False)

    pick_i = None
    if pickup_xy is not None:
        for i, p in enumerate(kin):
            if (int(p[0]), int(p[1])) == pickup_xy:
                pick_i = i
                break
    if pick_i is None:
        pick_i = -1  # already past pickup (delivery replan)

    unload_i = None
    for i in range(pick_i + 1, len(kin)):
        if (int(kin[i][0]), int(kin[i][1])) in end_set:
            unload_i = i
            break
    if unload_i is not None:
        # Consume in-place waits on the pad; unload on last pad tick before leave.
        pad = (int(kin[unload_i][0]), int(kin[unload_i][1]))
        while (
            unload_i + 1 < len(kin)
            and (int(kin[unload_i + 1][0]), int(kin[unload_i + 1][1])) == pad
        ):
            unload_i += 1

    if unload_i is None:
        # No pad on path. Never stamp FALSE on a non-pad terminus — that
        # created premature_unload at foreign hubs (e.g. Shanghai leave while
        # dest=Beijing). Keep carrying until a real pad appears / replan.
        if pick_i < 0:
            if end_set and (int(kin[-1][0]), int(kin[-1][1])) not in end_set:
                rows = path_to_step(kin, task, True, expand=False)
                for s in rows:
                    s["loaded"] = "TRUE"
                    s["destination"] = task.get("destination") or s.get("destination") or ""
                    s["task-id"] = task.get("task_id") or s.get("task-id") or ""
                    if task.get("priority", "Normal") != "Normal":
                        s["Emergency"] = "TRUE"
                return rows
            return path_to_step(kin, task, True, expand=False)
        pre = kin[: pick_i + 1]
        post = kin[pick_i + 1 :]
        out = path_to_step(pre, task, False, expand=False)
        if post:
            if end_set and (int(post[-1][0]), int(post[-1][1])) not in end_set:
                mid_rows = path_to_step(post, task, True, expand=False)
                for s in mid_rows:
                    s["loaded"] = "TRUE"
                    s["destination"] = task.get("destination") or s.get("destination") or ""
                    s["task-id"] = task.get("task_id") or s.get("task-id") or ""
                    if task.get("priority", "Normal") != "Normal":
                        s["Emergency"] = "TRUE"
                out.extend(mid_rows)
            else:
                out.extend(path_to_step(post, task, True, expand=False))
        return out

    out = []
    if pick_i >= 0:
        out.extend(path_to_step(kin[: pick_i + 1], task, False, expand=False))
        mid = kin[pick_i + 1 : unload_i + 1]
    else:
        mid = kin[: unload_i + 1]
    if mid:
        out.extend(path_to_step(mid, task, True, expand=False))
    leave = kin[unload_i + 1 :]
    if leave:
        out.extend(path_to_step(leave, task, False, expand=False))
    return out


def path_to_step(path, task, loaded, *, expand=None):
    if expand is None:
        expand = str(os.environ.get("AGV_EXPAND_TURN_MOVES", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if expand:
        path = expand_path_turn_moves(path)
    steps = []
    if not loaded:
        for i in range(len(path)):
            steps.append({
                "timestamp": path[i][2],
                "name": task['agv'],
                "X": path[i][0],
                "Y": path[i][1],
                "pitch": path[i][3],
                "loaded": "FALSE",
                "destination": "",
                "Emergency": "FALSE",
                "task-id": ""
            })
    else:
        if not path:
            return steps
        # Carry TRUE through every pose except the final unload pose (must be path[-1],
        # typically the extra wait tick at the dropoff). Using path[-2] here previously
        # marked mid-route / pickup cells as unloaded (Hound Dragon-2 @ station glitch).
        for i in range(len(path) - 1):
            steps.append({
                "timestamp": path[i][2],
                "name": task['agv'],
                "X": path[i][0],
                "Y": path[i][1],
                "pitch": path[i][3],
                "loaded": "TRUE",
                "destination": task['destination'],
                "Emergency": "FALSE" if task['priority'] == "Normal" else "TRUE",
                "task-id": task['task_id']
            })
        last = path[-1]
        steps.append({
            "timestamp": last[2],
            "name": task['agv'],
            "X": last[0],
            "Y": last[1],
            "pitch": last[3],
            "loaded": "FALSE",
            "destination": '',
            "Emergency": 'FALSE',
            "task-id": ''
        })
    return steps


def plan_relocation(task, start, goal, obstacles, moving_obstacles, grid_size=DEFAULT_GRID_SIZE,
                   wall_budget_s=None, max_visited=None, max_path_len=60, obstacle_horizon=None):
    """空载驶离到目标点（用于卸货点疏散），不含取卸货动作。"""
    if wall_budget_s is None:
        wall_budget_s = ASTAR_WALL_BUDGET_S
    if max_visited is None:
        max_visited = ASTAR_MAX_VISITED
    mo_h = int(obstacle_horizon if obstacle_horizon is not None else MOVING_OBSTACLE_HORIZON)
    plan_t0 = int(start[2]) if start and len(start) > 2 else 0
    t_deadline = _time.perf_counter() + max(0.05, float(wall_budget_s))
    _begin_astar_search(start[2] if start and len(start) > 2 else 0)

    def has_temp(pos, timestamp, temp_obstacle_positions):
        for _, temp in temp_obstacle_positions.items():
            if isinstance(temp, tuple) and len(temp) >= 2:
                temp_pos, temp_time = temp
                if temp_pos == pos and timestamp + 1 >= temp_time:
                    return True
        return False

    def get_valid_neighbors(state, temp_obstacle_positions):
        x, y, timestamp, direction = state
        moving_obstacle_positions = []
        turning_obstacle_positions = []
        for name, pos in moving_obstacles.items():
            if name == task['agv'] or not pos:
                continue
            # Planned poses always block; freeze-past-end only inside horizon.
            p = mo_occupied_cell(pos, timestamp + 1, plan_t0, mo_h)
            if p is not None and p not in moving_obstacle_positions:
                moving_obstacle_positions.append(p)
            p2 = mo_occupied_cell(pos, timestamp + 2, plan_t0, mo_h)
            if p2 is not None and p2 not in turning_obstacle_positions:
                turning_obstacle_positions.append(p2)

        possible_moves = [
            (1, 0, 0),
            (-1, 0, 180),
            (0, 1, 90),
            (0, -1, 270)
        ]
        if (x, y) not in obstacles and (x, y) not in moving_obstacle_positions:
            if not has_temp((x, y), timestamp, temp_obstacle_positions):
                yield (x, y, timestamp + 1, direction), 1, None

        for dx, dy, new_direction in possible_moves:
            new_x, new_y = x + dx, y + dy
            if not (1 <= new_x <= grid_size[0] and 1 <= new_y <= grid_size[1]):
                continue
            if direction == new_direction:
                if ((new_x, new_y) not in obstacles and
                    (new_x, new_y) not in moving_obstacle_positions and
                    not judge_swapping(moving_obstacles, (x, y, timestamp, direction),
                                       (new_x, new_y, timestamp + 1, new_direction), task['agv'])):
                    if not has_temp((new_x, new_y), timestamp, temp_obstacle_positions):
                        yield (new_x, new_y, timestamp + 1, new_direction), 1, None
            else:
                # main-copy：转向 + 行驶固定 2 个时间步、一次 yield 一个 neighbor。
                # 中间朝向不入 frontier；180° 的两拍 ±90° 由 expand_path_turn_moves 展开。
                if ((x, y) not in obstacles and (x, y) not in moving_obstacle_positions) and \
                   ((new_x, new_y) not in obstacles and (new_x, new_y) not in turning_obstacle_positions) and \
                   not judge_swapping(moving_obstacles, (x, y, timestamp + 1, direction),
                                      (new_x, new_y, timestamp + 2, new_direction), task['agv']):
                    if (not has_temp((x, y), timestamp, temp_obstacle_positions) and
                        not has_temp((new_x, new_y), timestamp + 1, temp_obstacle_positions)):
                        turn_state = (x, y, timestamp + 1, new_direction)
                        move_state = (new_x, new_y, timestamp + 2, new_direction)
                        yield move_state, 2, turn_state

    temp_obstacle_positions = {}
    for name, pos in moving_obstacles.items():
        if task['agv'] != name and pos:
            temp_obstacle_positions[name] = (pos[-1][:2], pos[-1][2])

    frontier = []
    visited = set()
    heapq.heappush(frontier, (astar_heuristic(start[:2], goal), 0, start, [start]))
    path = []
    while frontier:
        if _time.perf_counter() > t_deadline:
            path = []
            break
        _, cost, current, path = heapq.heappop(frontier)
        if len(path) > max_path_len or len(visited) > max_visited:
            path = []
            break
        if current[:2] == goal:
            break
        current_state = current[:3]
        if current_state in visited:
            continue
        visited.add(current_state)
        for neighbor, move_cost, turn_state in get_valid_neighbors(current, temp_obstacle_positions):
            neighbor_state = neighbor[:3]
            if neighbor_state in visited:
                continue
            inflate = conflict_move_inflate(
                neighbor[:2], neighbor[3], current[2], moving_obstacles, task["agv"]
            )
            new_cost = cost + move_cost * inflate
            predicted = new_cost + astar_heuristic(neighbor[:2], goal)
            # turn_state 只拼进路径；frontier 只压 move 终点（main-copy 范式）
            new_path = path + ([turn_state] if turn_state else []) + [neighbor]
            heapq.heappush(frontier, (predicted, new_cost, neighbor, new_path))

    if not path or path[-1][:2] != goal:
        return [], []

    steps = path_to_step(path, task, False)
    if path[0][2] != 0:
        return path[1:], steps[1:]
    return path, steps


def plan_clearout_after_unload(task, start, end_point_set, obstacles, moving_obstacles, grid_size=DEFAULT_GRID_SIZE):
    """卸货后驶离卸料位一格，释放卸货点；目标避开他人路径终点，并禁止对穿。"""
    static = set(tuple(p) if not isinstance(p, tuple) else p for p in obstacles)
    x0, y0, t0, d0 = start[0], start[1], start[2], start[3]
    finals = set()
    for name, traj in moving_obstacles.items():
        if name != task['agv'] and traj:
            finals.add(traj[-1][:2])

    def busy(pos, t):
        for name, traj in moving_obstacles.items():
            if name == task['agv'] or not traj:
                continue
            if t < len(traj) and traj[t][:2] == pos:
                return True
            if t >= len(traj) and traj[-1][:2] == pos:
                return True
        return False

    def would_swap(from_pos, to_pos, t_from, t_to):
        if from_pos == to_pos:
            return False
        for name, traj in moving_obstacles.items():
            if name == task['agv'] or not traj:
                continue
            a = traj[t_from][:2] if t_from < len(traj) else traj[-1][:2]
            b = traj[t_to][:2] if t_to < len(traj) else traj[-1][:2]
            if a == to_pos and b == from_pos:
                return True
        return False

    pitch_of = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270}
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        cand = (x0 + dx, y0 + dy)
        if not (1 <= cand[0] <= grid_size[0] and 1 <= cand[1] <= grid_size[1]):
            continue
        if cand in static or cand in end_point_set or cand in finals:
            continue
        new_dir = pitch_of[(dx, dy)]
        if d0 == new_dir:
            if busy(cand, t0 + 1) or would_swap((x0, y0), cand, t0, t0 + 1):
                continue
            return [start, (cand[0], cand[1], t0 + 1, new_dir)]
        if busy((x0, y0), t0 + 1) or busy(cand, t0 + 2):
            continue
        if would_swap((x0, y0), (x0, y0), t0, t0 + 1):
            continue
        if would_swap((x0, y0), cand, t0 + 1, t0 + 2):
            continue
        return [start, (x0, y0, t0 + 1, new_dir), (cand[0], cand[1], t0 + 2, new_dir)]
    return []


def _pos_at_traj(trajectory, t):
    """Lookup other AGV cell at time t (timestamp-aware; freeze past end)."""
    return mo_occupied_cell(trajectory, int(t), plan_t0=0, horizon=0)


def judge_swapping(agv_trajectories, current_pos_time_dir, new_pos_time_dir, self_name=None):
    current_x, current_y, current_t, _ = current_pos_time_dir
    new_x, new_y, new_t, _ = new_pos_time_dir
    if (current_x, current_y) == (new_x, new_y):
        return False
    # 仅检查相邻时刻的位置互换
    if new_t != current_t + 1:
        return False
    for name, trajectory in agv_trajectories.items():
        if self_name and name == self_name:
            continue
        if not trajectory:
            continue
        their_cur = _pos_at_traj(trajectory, current_t)
        their_new = _pos_at_traj(trajectory, new_t)
        if their_cur == (new_x, new_y) and their_new == (current_x, current_y):
            return True
    return False     
        
        
def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    
    
def A_Star(task, start: tuple, pickup_point: tuple, end_points: tuple, obstacles: list, moving_obstacles: dict = {}, grid_size: tuple = DEFAULT_GRID_SIZE,
           max_path_len=None, max_visited=None, max_frontier=None, wall_budget_s=None, obstacle_horizon=None):
    if max_path_len is None:
        max_path_len = ASTAR_MAX_PATH_LEN
    if max_visited is None:
        max_visited = ASTAR_MAX_VISITED
    if max_frontier is None:
        max_frontier = ASTAR_MAX_FRONTIER
    if wall_budget_s is None:
        wall_budget_s = ASTAR_WALL_BUDGET_S
    mo_h = int(obstacle_horizon if obstacle_horizon is not None else MOVING_OBSTACLE_HORIZON)
    plan_t0 = int(start[2]) if start and len(start) > 2 else 0
    t_deadline = _time.perf_counter() + max(0.05, float(wall_budget_s))
    _begin_astar_search(start[2] if start and len(start) > 2 else 0)
    
    def has_temp(pos, timestamp, temp_obstacle_positions):
        has_temp = False
        for agv_name, temp in temp_obstacle_positions.items():
            # print(f'{agv_name}:{temp}')
            if isinstance(temp, tuple) and len(temp) >= 2:
                temp_pos, temp_time = temp
                if temp_pos == pos and timestamp + 1 >= temp_time:
                    has_temp = True
                    break  # 找到后提前退出循环
        return has_temp
    
    def get_valid_neighbors(state,temp_obstacle_positions):
        """获取当前状态的有效邻居节点"""
        x, y, timestamp, direction = state
        moving_obstacle_positions = []
        turning_obstacle_positions = []

        for name, pos in moving_obstacles.items():
            if name == task['agv'] or not pos:
                continue
            # Planned poses always block; freeze-past-end only inside horizon.
            cell = mo_occupied_cell(pos, timestamp + 1, plan_t0, mo_h)
            if cell is not None and cell not in moving_obstacle_positions:
                moving_obstacle_positions.append(cell)
            cell2 = mo_occupied_cell(pos, timestamp + 2, plan_t0, mo_h)
            if cell2 is not None and cell2 not in turning_obstacle_positions:
                turning_obstacle_positions.append(cell2)
        possible_moves = [
            (1, 0, 0),    # 右
            (-1, 0, 180), # 左
            (0, 1, 90),   # 上
            (0, -1, 270)  # 下
        ]
        
        
        # 检查原地等待: 1. 不在障碍物中 2. 不在移动障碍物中 
        if (x,y) not in obstacles and (x,y) not in moving_obstacle_positions:
            if not has_temp((x,y),timestamp,temp_obstacle_positions):
                yield (x, y, timestamp+1, direction), 1, None
            
        # 检查移动和转向: 1. 不在障碍物中 2. 不在移动障碍物中 3. 不对穿
        for dx, dy, new_direction in possible_moves:
            new_x, new_y = x + dx, y + dy
            if not (1 <= new_x <= grid_size[0] and 1 <= new_y <= grid_size[1]):
                continue
            
            # 同方向移动
            if direction == new_direction:
                if ((new_x, new_y) not in obstacles and 
                    (new_x, new_y) not in moving_obstacle_positions and 
                    not judge_swapping(moving_obstacles,(x,y,timestamp,direction),(new_x,new_y,timestamp+1,new_direction), task['agv'])):
                    if not has_temp((new_x,new_y),timestamp,temp_obstacle_positions):
                        yield (new_x, new_y, timestamp+1, new_direction), 1, None
            # 需要转向：main-copy 固定 2 时间步（转向+行驶）一次 yield；不拆搜索状态
            else:
                if ((x, y) not in obstacles and (x, y) not in moving_obstacle_positions) and \
                    ((new_x, new_y) not in obstacles and (new_x, new_y) not in turning_obstacle_positions) and \
                    not judge_swapping(moving_obstacles,(x,y,timestamp+1,direction),(new_x,new_y,timestamp+2,new_direction), task['agv']):
                        if not has_temp((x,y),timestamp,temp_obstacle_positions) and not has_temp((new_x,new_y),timestamp+1,temp_obstacle_positions):
                            turn_state = (x, y, timestamp+1, new_direction)
                            move_state = (new_x, new_y, timestamp+2, new_direction)
                            yield move_state, 2, turn_state

        
        
    temp_obstacle_positions = {}
    for name, pos in moving_obstacles.items():
        if name not in temp_obstacle_positions and task['agv'] != name:
            temp_obstacle_positions[name] = []
        if task['agv'] != name:
            # 只有当路径不为空时才添加临时障碍物
            if pos and len(pos) > 0:
                if mo_h > 0:
                    best = pos[0]
                    for p in pos:
                        if int(p[2]) <= plan_t0 + mo_h:
                            best = p
                        else:
                            break
                    temp_obstacle_positions[name] = (best[:2], best[2])
                else:
                    temp_obstacle_positions[name] = (pos[-1][:2], pos[-1][2])

    # print(f'temp_obstacle_positions:{temp_obstacle_positions}')
    # print(f"当前 AGV {task['agv']} 起始位置: {start}")
    # print(f"目标取货点: {pickup_point}")
    # print(f"可用卸货点: {end_points}")
    # A*搜索主循环,从起点到供货台
    path = []
    steps = []
    frontier = []
    visited = set()
    loaded = 0
    # print(f'start:{start}')
    heapq.heappush(frontier, (astar_heuristic(start[:2], pickup_point), 0, start, [start]))
    
    while frontier:
        if _time.perf_counter() > t_deadline:
            path = []
            break
        _, cost, current, path = heapq.heappop(frontier)
        
        if len(path) > max_path_len or len(visited) > max_visited or len(frontier) > max_frontier:
            path = []
            # print('太难找了')
            # print(pickup_point)
            # print(frontier)
            break
            
        # 到达供货台
        if current[:2] == pickup_point:
            pickup_movement = (*pickup_point[:2],current[2]+1,current[3])
            steps.extend(path_to_step(path,task,loaded))

            
            loaded = 1
            break
            
        # 检查访问状态
        current_state = current[:3]
        if current_state in visited:
            continue
        visited.add(current_state)
        
        # 扩展邻居节点（turn_state 只拼路径，不单独入堆）
        for neighbor, move_cost, turn_state in get_valid_neighbors(current,temp_obstacle_positions):
            neighbor_state = neighbor[:3]
            if neighbor_state in visited:
                continue
                
            inflate = conflict_move_inflate(
                neighbor[:2], neighbor[3], current[2], moving_obstacles, task["agv"]
            )
            new_cost = cost + move_cost * inflate
            predicted_cost = new_cost + astar_heuristic(neighbor[:2], pickup_point)
            
            new_path = path + ([turn_state] if turn_state else []) + [neighbor]
            heapq.heappush(frontier, (predicted_cost, new_cost, neighbor, new_path))
        
        
        
        
        
    if not path or (path[-1][0], path[-1][1]) != pickup_point:
        return [], []

    # Assign-time: plan only start→pickup; delivery is replanned after goods are on board
    # (AGV_PICKUP_THEN_DELIVERY). Delivery replan sets start==pickup so full second leg runs.
    _ptd = str(os.environ.get("AGV_PICKUP_THEN_DELIVERY", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _force_full = bool(isinstance(task, dict) and task.get("force_full_path"))
    if (
        _ptd
        and not _force_full
        and tuple(start[:2]) != tuple(pickup_point[:2])
        and end_points
    ):
        path = list(path) + [pickup_movement]
        steps.append(
            {
                "timestamp": pickup_movement[2],
                "name": task["agv"],
                "X": pickup_movement[0],
                "Y": pickup_movement[1],
                "pitch": pickup_movement[3],
                "loaded": "TRUE",
                "destination": task.get("destination") or "",
                "Emergency": "FALSE"
                if task.get("priority", "Normal") == "Normal"
                else "TRUE",
                "task-id": task["task_id"],
            }
        )
        if path[0][2] != 0:
            return path[1:], steps[1:]
        return path, steps

    #从供货台到卸货点：优先空闲卸料位，被占的也尝试（可等待），卸完立刻驶离让位
    free_ends, busy_ends = [], []
    for ep in list(end_points):
        occupied = False
        for temp in temp_obstacle_positions.values():
            if isinstance(temp, tuple) and len(temp) >= 2:
                if temp[0] == ep and pickup_movement[2] + 1 > temp[1]:
                    occupied = True
                    break
        (busy_ends if occupied else free_ends).append(ep)

    possible_paths = []
    for end_point in free_ends + busy_ends:
        if _time.perf_counter() > t_deadline:
            break
        frontier = []
        visited = set()
        start_state = pickup_movement
        heapq.heappush(frontier, (astar_heuristic(start_state[:2], end_point), 0, start_state, [start_state]))
        possible_path = []
        while frontier:
            if _time.perf_counter() > t_deadline:
                possible_path = []
                break
            _, cost, current, possible_path = heapq.heappop(frontier)
            if len(possible_path) > max_path_len or len(visited) > max_visited or len(frontier) > max_frontier:
                possible_path = []
                break
            if current[:2] == end_point:
                possible_paths.append(possible_path)
                break
            current_state = current[:3]
            if current_state in visited:
                continue
            visited.add(current_state)
            for neighbor, move_cost, turn_state in get_valid_neighbors(current, temp_obstacle_positions):
                neighbor_state = neighbor[:3]
                if neighbor_state in visited:
                    continue
                inflate = conflict_move_inflate(
                    neighbor[:2], neighbor[3], current[2], moving_obstacles, task["agv"]
                )
                new_cost = cost + move_cost * inflate
                predicted_cost = new_cost + astar_heuristic(neighbor[:2], end_point)
                new_path = possible_path + ([turn_state] if turn_state else []) + [neighbor]
                heapq.heappush(frontier, (predicted_cost, new_cost, neighbor, new_path))

    min_cost = float('inf')
    min_path = []
    for p in possible_paths:
        # 卸货占用窗口 [到达, 到达+1] 不可与他人重叠
        arrive_t = p[-1][2]
        ep = p[-1][:2]
        conflict = False
        for name, traj in moving_obstacles.items():
            if name == task['agv'] or not traj:
                continue
            for tt in (arrive_t, arrive_t + 1):
                if tt < len(traj) and traj[tt][:2] == ep:
                    conflict = True
                    break
                if tt >= len(traj) and traj[-1][:2] == ep:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        if len(p) < min_cost:
            min_cost = len(p)
            min_path = p

    if min_path == []:
        return [], []

    # 卸货停留 1s
    min_path = min_path + [(min_path[-1][0], min_path[-1][1], min_path[-1][2] + 1, min_path[-1][3])]
    steps.extend(path_to_step(min_path, task, loaded))
    path.extend(min_path)

    # 卸完就近驶离卸料位，释放格口给后续车辆
    clear_path = plan_clearout_after_unload(
        task, path[-1], set(end_points), obstacles, moving_obstacles, grid_size
    )
    if clear_path:
        if clear_path[0][:3] == path[-1][:3]:
            clear_body = clear_path[1:]
        else:
            clear_body = clear_path
        if clear_body:
            steps.extend(path_to_step(clear_body, task, False))
            path.extend(clear_body)

    if path[0][2] != 0:
        return path[1:], steps[1:]
    return path, steps
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    

    
    
    
if __name__ == '__main__':
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    _DATA = _ROOT / "data"
    _TRAJ = _DATA / "outputs" / "trajectories"

    AGV_POSITION_PATH = str(_DATA / "agv_position.csv")
    AGV_TASK_PATH = str(_DATA / "agv_task_1.csv")  # 常规: agv_task.csv；极端同卸货点: agv_task_1.csv
    AGV_TRAJECTORY_PATH = str(_TRAJ / "agv_trajectory_extreme.csv")  # normal/prob 变体同名目录
    #获取所有任务
    all_tasks = get_task_list(AGV_TASK_PATH)
    #获取取货点、卸货点和小车初始状态
    start_points, end_points, agv_list = get_object_position(AGV_POSITION_PATH)

    #静态障碍
    env =  ENV(list(start_points.values()),list(end_points.values()))
    
    #补充小车信息
    agv_states = get_agv_state(agv_list)
    #整理任务状态
    task_states = get_task_state(start_points,all_tasks,end_points)
    unassigned_agvs = list(agv_states.keys())
    # print(agv_states)
    # print(task_states)

    # 清空旧轨迹
    with open(AGV_TRAJECTORY_PATH, 'w', newline='') as f:
        pass

    sim = Simulation(agv_states,task_states,env,DEFAULT_GRID_SIZE)
    while not sim.all_over():
        sim.time_forward()
        if sim.time % 50 == 0:
            left = sum(len(v) for v in sim.task_states.values())
            print(f"[progress] t={sim.time}, tasks_left={left}, surface={list(sim.surface_tasks.keys())}")
        if sim.time > 5000:
            print("[error] 超过最大仿真时长，强制结束")
            break
        
    all_steps = sim.steps_reorganize()
    print(f"完成时刻: {sim.time}, 步数条目: {len(all_steps)}")
    
    append_to_csv(all_steps)
    
    check_trajectory_conflicts(AGV_TRAJECTORY_PATH)
    summarize_utilization(AGV_TRAJECTORY_PATH)