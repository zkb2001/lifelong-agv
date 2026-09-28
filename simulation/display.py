import pygame
import csv
import math
import time
from collections import defaultdict
import sys
import numpy as np
import importlib

imageio = None

# 初始化pygame
pygame.init()

# 常量定义
GRID_SIZE = 40  # 每个格子的大小
WINDOW_SIZE = (20 * GRID_SIZE, 20 * GRID_SIZE)  # 修改为20x20
FPS = 3  # 每秒更新一次（视频/预览里每秒播放多少仿真步）

# 颜色定义
WHITE = (255, 255, 255)
BLUE = (100, 149, 237)  # 供货台颜色
GREEN = (50, 205, 50)   # 卸货点颜色
BLACK = (0, 0, 0)
RED = (255, 0, 0)       # 时间显示和紧急任务
PINK = (255, 192, 203)  # 普通任务
GRAY = (128, 128, 128)  # 网格线

# AGV颜色列表 - 16种不同的浅色
AGV_COLORS = [
    (255, 182, 193),  # 浅粉红
    (255, 218, 185),  # 桃色
    (255, 250, 205),  # 柠檬雪纺
    (176, 224, 230),  # 粉蓝
    (221, 160, 221),  # 梅红
    (216, 191, 216),  # 蓟色
    (240, 230, 140),  # 卡其布
    (238, 232, 170),  # 浅黄
    (152, 251, 152),  # 浅绿
    (135, 206, 250),  # 浅天蓝
    (230, 230, 250),  # 薰衣草
    (255, 239, 213),  # 番木瓜
    (255, 228, 225),  # 薄雾玫瑰
    (245, 245, 220),  # 米色
    (240, 248, 255),  # 爱丽丝蓝
    (250, 235, 215),  # 古董白
]

# 城市名称到编号的映射
CITY_TO_NUMBER = {
    'Hangzhou': 1, 'Guangzhou': 2, 'Urumqi': 3, 'Chongqing': 4,
    'Suzhou': 5, 'Changsha': 6, 'Kunming': 7, 'Tianjin': 8,
    'Shanghai': 9, 'Wuhan': 10, 'Xiamen': 11, 'Dalian': 12,
    'Beijing': 13, 'Nanjing': 14, 'Chengdu': 15, 'Shenzhen': 16
}

class DisplayManager:
    def __init__(self, speed=1, record=True, video_filename='agv_simulation.mp4',
                 task_file='agv_task.csv', trajectory_file='agv_trajectory.csv',
                 position_file='agv_position.csv', obstacle_cells=None,
                 video_duration_sec=None):
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("AGV Simulation")
        self.clock = pygame.time.Clock()
        self.speed = speed
        self.record = record
        self.video_filename = video_filename
        self.task_file = task_file
        self.trajectory_file = trajectory_file
        self.position_file = position_file
        self.obstacle_cells = set(obstacle_cells or [])
        self.video_duration_sec = video_duration_sec
        self.video_writer = None
        self.video_saved = False
        
        # 加载数据
        self.load_position_data()
        self.load_task_data()
        self.load_trajectory_data()
        
        # 初始化字体
        self.font = pygame.font.Font(None, 24)
        self.time_font = pygame.font.Font(None, 36)
        
        # 任务完成状态跟踪
        self.completed_tasks = set()  # 已取货（供货台剩余计数用）
        self.current_tasks = {}  # AGV -> task_id mapping
        self.remaining_tasks = self.tasks_per_start.copy()
        self.n_picked = 0
        self.n_delivered = 0  # 已卸货 = 真正完成的任务数
        self.n_tasks_total = len(self.task_info)
        
        # 新增：任务顺序跟踪
        self.task_sequence = {}  # start_point -> [task_ids] 按顺序排列
        self.current_task_index = {}  # start_point -> 当前任务索引
        self.initialize_task_sequence()
        
        # AGV颜色映射
        self.agv_colors = {}
        agv_list = sorted(list(self.agv_positions.keys()))
        for i, agv in enumerate(agv_list):
            self.agv_colors[agv] = AGV_COLORS[i % len(AGV_COLORS)]
            
        # 记录上一个时间戳的状态
        self.last_state = {}
        
        # 初始化视频写入器
        self.setup_video_writer()
    
    def setup_video_writer(self):
        """根据配置初始化视频写入器"""
        if not self.record:
            return
        global imageio
        try:
            if imageio is None:
                imageio = importlib.import_module('imageio.v2')
            # 默认 FPS=3、speed=1：每仿真步一帧。可用 --duration 将成片时长压到约 N 秒（不改跳帧逻辑）。
            if self.video_duration_sec and self.max_timestamp >= 0:
                fps = max(1, round((self.max_timestamp + 1) / float(self.video_duration_sec)))
            else:
                fps = max(1, int(FPS * self.speed))
            self._video_fps = fps
            self.video_writer = imageio.get_writer(
                self.video_filename,
                fps=fps,
                codec='libx264',
                macro_block_size=None
            )
        except Exception as exc:
            print(f"初始化视频写入器失败：{exc}")
            self.video_writer = None
    
    def capture_frame(self):
        """抓取当前屏幕帧并写入视频"""
        if not self.video_writer:
            return
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.video_writer.append_data(frame)
    
    def close_video_writer(self):
        """关闭视频写入器"""
        if self.video_writer:
            self.video_writer.close()
            self.video_writer = None
            self.video_saved = True
    
    def finalize_recording(self, show_message=False):
        """结束录制并按需提示"""
        if self.record and not self.video_saved:
            self.close_video_writer()
            if show_message:
                print(f"可视化已保存至 {self.video_filename}")

    def convert_y_coordinate(self, y):
        """转换Y坐标（从底部向上计数）"""
        return 19 - y  # 20x20网格，索引从0开始

    def load_position_data(self):
        """加载位置数据"""
        self.start_points = {}  # 供货台位置
        self.end_points = {}    # 卸货点位置
        self.agv_positions = {} # AGV初始位置和朝向
        
        with open(self.position_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row['x']) - 1  # 坐标从1开始，转换为从0开始
                y = int(row['y']) - 1
                pos = (x, y)
                if row['type'] == 'start_point':
                    self.start_points[row['name']] = pos
                elif row['type'] == 'end_point':
                    self.end_points[row['name']] = pos
                elif row['type'] == 'agv':
                    self.agv_positions[row['name']] = {
                        'pos': pos,
                        'pitch': int(row['pitch']) if row['pitch'] else 0
                    }

    def load_task_data(self):
        """加载任务数据"""
        self.tasks_per_start = defaultdict(int)  # 每个供货台的任务数
        self.urgent_tasks = defaultdict(bool)    # 每个供货台是否有紧急任务
        self.task_info = {}                      # 任务ID到任务信息的映射
        self.start_point_tasks = defaultdict(list)  # 每个供货台的任务列表
        
        # 读取所有任务并按start_point分组
        tasks_by_start = defaultdict(list)
        with open(self.task_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tasks_by_start[row['start_point']].append({
                    'task_id': row['task_id'],
                    'start_point': row['start_point'],
                    'end_point': row['end_point'],
                    'priority': row['priority']
                })
        
        # 对每个供货台的任务进行排序和处理
        for start_point, tasks in tasks_by_start.items():
            # 使用自然排序，确保Tiger-1在Tiger-10之前
            sorted_tasks = sorted(tasks, key=lambda x: self.natural_sort_key(x['task_id']))
            self.tasks_per_start[start_point] = len(sorted_tasks)
            
            # 保存排序后的任务ID列表
            self.start_point_tasks[start_point] = [task['task_id'] for task in sorted_tasks]
            
            # 保存任务信息
            for task in sorted_tasks:
                self.task_info[task['task_id']] = {
                    'start_point': task['start_point'],
                    'end_point': task['end_point'],
                    'priority': task['priority']
                }
                if task['priority'] == 'Urgent':
                    self.urgent_tasks[start_point] = True

    def natural_sort_key(self, task_id):
        """生成自然排序的键，例如 'Tiger-1' 会排在 'Tiger-10' 之前"""
        import re
        parts = re.split('([0-9]+)', task_id)
        parts[1:] = [int(num) if num.isdigit() else num for num in parts[1:]]
        return parts

    def load_trajectory_data(self):
        """加载轨迹数据，并按 timestamp 预索引（避免每帧全表扫描）。"""
        self.trajectory_data = []
        with open(self.trajectory_file, 'r') as f:
            reader = csv.DictReader(f)
            self.trajectory_data = list(reader)
        self.traj_by_t = defaultdict(dict)  # t -> {agv_name: row}
        for row in self.trajectory_data:
            t = int(row['timestamp'])
            self.traj_by_t[t][row['name']] = row
        self.max_timestamp = max(self.traj_by_t) if self.traj_by_t else -1

    def initialize_task_sequence(self):
        """初始化任务顺序跟踪"""
        for start_point in self.start_point_tasks:
            self.task_sequence[start_point] = self.start_point_tasks[start_point]
            self.current_task_index[start_point] = 0

    def get_next_task_for_start_point(self, start_point):
        """获取供货台的下一个未完成任务"""
        current_task_list = self.task_sequence[start_point]
        for task_id in current_task_list:
            if task_id not in self.completed_tasks and task_id not in self.current_tasks.values():
                return task_id
        return None

    def get_current_task_for_start_point(self, start_point):
        """获取供货台当前应该显示的任务（按顺序，跳过已取走的）。"""
        if start_point not in self.task_sequence:
            return None
        
        current_task_list = self.task_sequence[start_point]
        current_index = self.current_task_index[start_point]
        
        while current_index < len(current_task_list):
            tid = current_task_list[current_index]
            if tid not in self.completed_tasks:
                self.current_task_index[start_point] = current_index
                return tid
            current_index += 1
        
        self.current_task_index[start_point] = current_index
        return None

    def advance_task_for_start_point(self, start_point):
        """推进供货台的任务索引（当任务被取走时）"""
        if start_point in self.current_task_index:
            self.current_task_index[start_point] += 1

    def draw_obstacles(self):
        """绘制静态障碍（坐标与 position.csv / map JSON 一致，1-indexed）。"""
        for ox, oy in self.obstacle_cells:
            gx = int(ox) - 1
            gy = int(oy) - 1
            if not (0 <= gx < 20 and 0 <= gy < 20):
                continue
            x = gx * GRID_SIZE + 1
            y = self.convert_y_coordinate(gy) * GRID_SIZE + 1
            pygame.draw.rect(
                self.screen, (55, 55, 60), (x, y, GRID_SIZE - 2, GRID_SIZE - 2)
            )
            pygame.draw.rect(
                self.screen, (35, 35, 40), (x, y, GRID_SIZE - 2, GRID_SIZE - 2), 1
            )

    def draw_grid(self):
        """绘制网格"""
        for i in range(20):  # 改为20x20
            pygame.draw.line(self.screen, GRAY, (i * GRID_SIZE, 0), 
                           (i * GRID_SIZE, WINDOW_SIZE[1]))
            pygame.draw.line(self.screen, GRAY, (0, i * GRID_SIZE), 
                           (WINDOW_SIZE[0], i * GRID_SIZE))

    def draw_arrow(self, pos, pitch):
        """绘制AGV方向箭头"""
        x = pos[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.convert_y_coordinate(pos[1]) * GRID_SIZE + GRID_SIZE // 2
        # 调整角度：pygame的y轴向下为正，我们的坐标系y轴向上为正
        angle = math.radians(-pitch)  # 取负值来反转方向
        arrow_length = GRID_SIZE // 2
        end_x = x + arrow_length * math.cos(angle)
        end_y = y + arrow_length * math.sin(angle)
        pygame.draw.line(self.screen, BLACK, (x, y), (end_x, end_y), 2)
        
        # 绘制箭头头部
        head_length = 10
        angle_left = angle - math.pi / 6
        angle_right = angle + math.pi / 6
        left_x = end_x - head_length * math.cos(angle_left)
        left_y = end_y - head_length * math.sin(angle_left)
        right_x = end_x - head_length * math.cos(angle_right)
        right_y = end_y - head_length * math.sin(angle_right)
        pygame.draw.line(self.screen, BLACK, (end_x, end_y), (left_x, left_y), 2)
        pygame.draw.line(self.screen, BLACK, (end_x, end_y), (right_x, right_y), 2)

    def draw_cargo(self, pos, priority, end_point):
        """绘制货物"""
        x = pos[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.convert_y_coordinate(pos[1]) * GRID_SIZE + GRID_SIZE // 2
        color = RED if priority == 'Urgent' else PINK
        pygame.draw.circle(self.screen, color, (x, y), GRID_SIZE // 3)
        
        # 绘制目的地编号
        number = CITY_TO_NUMBER[end_point]
        text = self.font.render(str(number), True, BLACK)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)

    def draw_remaining_tasks(self, start_point, count, has_urgent):
        """绘制剩余任务数"""
        pos = self.start_points[start_point]
        x = pos[0] * GRID_SIZE + GRID_SIZE // 2
        y = self.convert_y_coordinate(pos[1]) * GRID_SIZE - 15
        text = str(count) + ('!!' if has_urgent else '')
        text_surface = self.font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)

    def draw_timestamp(self, timestamp):
        """绘制时间戳 + 实时完成任务数（卸货完成）。"""
        text = self.time_font.render(f"Time: {timestamp}", True, RED)
        self.screen.blit(text, (10, 10))
        total = self.n_tasks_total
        if total > 0:
            done_line = f"Completed: {self.n_delivered} / {total}"
        else:
            done_line = f"Completed: {self.n_delivered}"
        done_surf = self.time_font.render(done_line, True, RED)
        self.screen.blit(done_surf, (10, 46))
        pick_surf = self.font.render(
            f"Picked: {self.n_picked}  Active: {len(self.current_tasks)}",
            True,
            BLACK,
        )
        self.screen.blit(pick_surf, (10, 82))

    def update_task_status(self, current_state, prev_state):
        """更新任务状态"""
        for agv_name, row in current_state.items():
            loaded = row['loaded'].lower() == 'true'
            task_id = (row.get('task-id') or '').strip()
            
            # 获取上一个状态
            prev_loaded = False
            prev_tid = ''
            if agv_name in prev_state:
                prev_loaded = prev_state[agv_name]['loaded'].lower() == 'true'
                prev_tid = (prev_state[agv_name].get('task-id') or '').strip()
            
            # 如果AGV装载了新货物（从unloaded变为loaded）
            if not prev_loaded and loaded and task_id:
                # 记录当前任务
                self.current_tasks[agv_name] = task_id
                # 减少对应供货台的剩余任务数
                if task_id not in self.completed_tasks:
                    info = self.task_info.get(task_id)
                    if info:
                        start_point = info['start_point']
                        self.remaining_tasks[start_point] -= 1
                        self.completed_tasks.add(task_id)
                        self.n_picked += 1
                        # 推进该供货台的任务索引
                        self.advance_task_for_start_point(start_point)
            
            # 如果AGV卸载了货物（从loaded变为unloaded）→ 计为完成
            elif prev_loaded and not loaded:
                done_tid = self.current_tasks.pop(agv_name, None) or prev_tid
                if done_tid:
                    self.n_delivered += 1

    def draw_frame(self, timestamp):
        """绘制单个时间点的画面"""
        self.screen.fill(WHITE)
        self.draw_obstacles()
        self.draw_grid()
        
        # 绘制时间戳
        self.draw_timestamp(timestamp)
        
        # 获取当前 / 上一时刻状态（预索引）
        current_state = self.traj_by_t.get(timestamp, {})
        prev_timestamp = timestamp - 1 if timestamp > 0 else 0
        prev_state = self.traj_by_t.get(prev_timestamp, {})
        
        # 更新任务状态
        self.update_task_status(current_state, prev_state)
        
        # 绘制供货台和卸货点
        for name, pos in self.start_points.items():
            pygame.draw.rect(self.screen, BLUE, 
                           (pos[0] * GRID_SIZE, 
                            self.convert_y_coordinate(pos[1]) * GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
            
            # 在供货台绘制当前应该显示的任务（按顺序）
            current_task_id = self.get_current_task_for_start_point(name)
            if current_task_id:
                task = self.task_info[current_task_id]
                self.draw_cargo(pos, task['priority'], task['end_point'])
            
        for name, pos in self.end_points.items():
            pygame.draw.rect(self.screen, GREEN, 
                           (pos[0] * GRID_SIZE, 
                            self.convert_y_coordinate(pos[1]) * GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
            number = CITY_TO_NUMBER[name]
            text = self.font.render(str(number), True, BLACK)
            text_rect = text.get_rect(center=(pos[0] * GRID_SIZE + GRID_SIZE // 2, 
                                            self.convert_y_coordinate(pos[1]) * GRID_SIZE + GRID_SIZE // 2))
            self.screen.blit(text, text_rect)
        
        # 更新任务状态和绘制AGV
        for agv_name, row in current_state.items():
            x, y = int(row['X']) - 1, int(row['Y']) - 1  # 转换坐标
            # 仓库可视区为 0..19；轨迹若越界则跳过（避免画面外闪现）
            if not (0 <= x < 20 and 0 <= y < 20):
                continue
            pitch = int(row['pitch'])
            loaded = row['loaded'].lower() == 'true'
            task_id = row.get('task-id', '')
            
            # 绘制AGV（使用独特的颜色）
            pygame.draw.rect(self.screen, self.agv_colors[agv_name], 
                           (x * GRID_SIZE, 
                            self.convert_y_coordinate(y) * GRID_SIZE, 
                            GRID_SIZE, GRID_SIZE))
            
            # 如果AGV载有货物，绘制货物
            if loaded and (task_id in self.task_info or agv_name in self.current_tasks):
                task_id = task_id or self.current_tasks[agv_name]
                task = self.task_info[task_id]
                self.draw_cargo((x, y), task['priority'], task['end_point'])
            
            # 最后绘制箭头（确保在货物上方）
            self.draw_arrow((x, y), pitch)
        
        # 绘制剩余任务数
        for start_point in self.remaining_tasks:
            has_urgent = any(
                self.task_info[task_id]['priority'] == 'Urgent'
                for task_id in self.task_info
                if self.task_info[task_id]['start_point'] == start_point
                and task_id not in self.completed_tasks
            )
            self.draw_remaining_tasks(start_point, self.remaining_tasks[start_point], has_urgent)
        
        pygame.display.flip()
        self.capture_frame()

    def run(self):
        """运行显示"""
        current_timestamp = 0
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.finalize_recording(show_message=True)
                    running = False
            
            if current_timestamp <= self.max_timestamp:
                self.draw_frame(current_timestamp)
                current_timestamp += 1
                # 录制时不按墙钟节流（8000s 轨迹否则会等数十分钟）
                if not self.record:
                    self.clock.tick(FPS * self.speed)
                elif current_timestamp % 500 == 0:
                    print(
                        f"[display] t={current_timestamp}/{self.max_timestamp} "
                        f"completed={self.n_delivered}",
                        flush=True,
                    )
            else:
                # 播完自动结束，便于无人值守录制
                self.finalize_recording(show_message=True)
                running = False
            
            pygame.display.flip()
        
        pygame.quit()
        self.finalize_recording()

if __name__ == "__main__":
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    _DATA = _ROOT / "data"
    _TRAJ = _DATA / "outputs" / "trajectories"
    _VID = _DATA / "outputs" / "videos"

    speed = 1  # 默认速度
    record = True
    video_filename = str(_VID / "agv_simulation.mp4")
    task_file = str(_DATA / "agv_task.csv")
    trajectory_file = str(_TRAJ / "agv_trajectory.csv")
    position_file = str(_DATA / "agv_position.csv")
    obstacle_file = None
    video_duration_sec = None
    
    args = sys.argv[1:]
    
    # 解析速度参数（如果第一个参数是数字）
    if args:
        try:
            speed = float(args[0])
            args = args[1:]
        except ValueError:
            pass
    
    # 解析录制参数
    if '--no-record' in args:
        record = False
    elif '--record' in args:
        idx = args.index('--record')
        if idx + 1 < len(args) and not args[idx + 1].startswith('--'):
            video_filename = args[idx + 1]
    
    if '--task' in args:
        idx = args.index('--task')
        if idx + 1 < len(args):
            task_file = args[idx + 1]
    if '--traj' in args:
        idx = args.index('--traj')
        if idx + 1 < len(args):
            trajectory_file = args[idx + 1]
    if '--position' in args:
        idx = args.index('--position')
        if idx + 1 < len(args):
            position_file = args[idx + 1]
    if '--obstacles' in args:
        idx = args.index('--obstacles')
        if idx + 1 < len(args):
            obstacle_file = args[idx + 1]
    if '--duration' in args:
        idx = args.index('--duration')
        if idx + 1 < len(args):
            try:
                video_duration_sec = float(args[idx + 1])
            except ValueError:
                video_duration_sec = None

    obstacle_cells = []
    if obstacle_file:
        import json
        with open(obstacle_file, encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            pts = raw.get('extra_obstacles') or raw.get('obstacles') or []
        elif isinstance(raw, list):
            pts = raw
        else:
            pts = []
        for p in pts:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                obstacle_cells.append((int(p[0]), int(p[1])))
        if obstacle_cells:
            print(
                f"[display] loaded {len(obstacle_cells)} obstacle cells from {obstacle_file}",
                flush=True,
            )
    
    display = DisplayManager(
        speed, record, video_filename, task_file, trajectory_file,
        position_file=position_file, obstacle_cells=obstacle_cells,
        video_duration_sec=video_duration_sec,
    )
    display.run() 