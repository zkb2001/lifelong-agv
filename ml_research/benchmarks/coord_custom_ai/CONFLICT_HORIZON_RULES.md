# Conflict-Horizon 规则（AI 依从版）

> 实现：`solve_conflict_horizon.py` · 校验：`validate_hybrid_trajectory.py` · 渲染：`simulation/display.py`  
> 竞赛运动语义：`competition/navigation.py`

---

## 1. 时间与坐标

| 概念 | 定义 |
|------|------|
| **sim_t** | 仿真秒 = 轨迹 `timestamp`；每步 +1；**1 仿真秒 = 视频 1 帧** |
| **wave** | 求解器重规划轮次；**≠ sim_t**；无冲突时可一次推进多秒 |
| **网格** | 1-indexed，x/y ∈ [1,20]；4 连通 |
| **Pose** | `(x, y, pitch)`；pitch ∈ {0,90,180,270} |

---

## 2. 运动（硬约束）

每 AGV 每 sim_t **至多一种原子动作**：

| 动作 | 耗时 | 效果 |
|------|------|------|
| **wait** | 1s | (x,y,pitch) 不变 |
| **turn** | 1s | (x,y) 不变，pitch → 目标朝向 |
| **move** | 1s | 沿当前 pitch 走 1 格，pitch 与移动方向一致 |

**转向后移动**：pitch ≠ 移动方向时，必须先 **turn 1s**，再 **move 1s**（共 2s）。**禁止**同秒变 pitch 且变 (x,y)。

**pitch 与方向**：+x→0，−x→180，+y→90，−y→270。

**禁止**：穿墙（`static` + `extra_obstacles`）、越界、曼哈顿步长 >1、同格占用（vertex）、对穿（swap）。

---

## 3. 冲突视界（核心算法）

### 3.1 小波 → 大波（车流融合）

- **每车一小波**：每台 AGV 初始 `wave_id = 自己`（micro-wave）。  
- **无时损融合**：同一 wave 内若各车 solo 路径在视界内无 vertex/swap，则融成更大波；融合后联合提交时长 = 各车 solo 时长（**禁止**为融合而整波空等）。  
- **冲突时**：
  1. 先尝试把仍兼容的车流再吸进较大一侧（`wave-absorb`）；  
  2. **车流更大的波**作 keeper（走廊优先）；  
  3. 较小波整组让位（Evacuate / yield），**尽最大努力前进但不进入大波走廊**；  
  4. 旁观者（非冲突组）继续阶段目标，禁止因一对死锁全体冻结。

### 3.2 每 wave 循环

每 **wave**：

1. **派单**：空闲 AGV 立即补满 `max_active`；**每台空车**对队列中全部任务算**单车 A\*** 代价（`pos→pickup + pickup→最近卸点`，忽略他车），取最小者分配。卸货完成的同一仿真秒也会立刻再派，**不等下一 wave**。  
2. **Solo 路径**：每车独立 BFS/A*  toward 当前阶段目标（`to_pickup` → 取货点；`to_drop` → 最近可卸点）。**规划时不考虑他车**。  
3. **展开时间线**：xy 路径 → 含 turn 的 Pose 时间线（§2）。  
4. **小波融合**：对当前 traj 做 `_fuse_waves_no_loss`（§3.1）。  
5. **叠加找 T\***：全员 Pose 时间线首次 **vertex** 或 **swap** 的 sim 索引。  
6. **提交**：
   - 无冲突 → 沿时间线推进；  
   - 有冲突 → 仅提交到 **T\*−1**，再让位（§3.1 大波优先）。  
   - **阶段事件打断**：任一 AGV 取货完成（缺 `to_drop` 路径）、卸货完成并即时派单（缺 `to_pickup` 路径）时，**立刻停止本次 commit**，下一循环重规划。不是按固定 \(h\) 秒滚动，而是「有车缺下一阶段路径就规划」。  
7. **让位**：低优先级 / 较小波改道；高优先级 / 较大波（keeper）继续。重复至 `validate_ok` 或 stall 上限。  
   - **旁观者不停**：仅冲突组参与让位/Evacuate；其余 AGV **与让位逐步交错**继续阶段目标（旧 wave 时间线用尽则 greedy 补一步）。禁止因一对死锁而全体原地等待。  
   - **多车死锁**：chronic/swap 时把挡在 corridor / Evacuate 退路上的第三车并入 yield 组；先清外侧挡路车，再清内侧，最后 keeper PUSH（2 车是 \|group\|=2 的特例）。

**idle 车**：无任务时向最近排队取货点 staging（`_idle_staging_goal`），参与叠加与让位，**禁止**整局静止。

---

## 4. 优先级（yield 排序，高者优先 / 低者让）

1. 非 `yield_until` 锁定期  
2. **所属波更大者优先**（融合后的车流规模）  
3. **Urgent** > Normal  
4. **loaded** > empty  
5. **remA\***（到当前 leg 目标）小者优先  
6. 车名 tie-break  

**慢性死锁**（同 pair `pair_hits ≥ 阈值` 或同一姿态连续 ≥3）：锁定 keeper（仍服从大波优先）；**对向 `swap`（尤其 t\*=1）** 时 yielder 执行 **Evacuate**（沿占用带退出，允许本步 remA\* 上升），禁止空 wait。仅当双方格子发生变化后才清零 `pair_hits`。

**停车哲学**：vertex 让位格是里程碑，优先 **remA\* 更小**；**swap / yield miss** 以离开 keeper 占用带为准，不要求本步 remA\* 下降。走廊清后立即 **resume**。

---

## 5. 任务状态机

```
assign → leg=to_pickup → 到 pickup（loaded=TRUE, leg=to_drop）→ 到 end_point（done, 释放 AGV）
```

- 每 AGV 同时最多 1 个 active 任务。  
- `end_points` 须在 `static` 外（solver 已过滤）。
- **取货交互格**：只能在取货台 **正左或正右**（同 y、x±1）；禁止正上/正下或更远邻格。
- **全员参与**：8 车都应接活（`work_count` 公平）；FIFO 下同时取货仍 ≤6 站，其余车应在送货/staging/候下一单，禁止长期空闲。
- **取货后分波进入**：取货完成后 **不必** 与同批车辆同步进入送货/离场；先到先送、先卸先领下一单（事件驱动），禁止整波 via-leave 同步等待。语义上等同 §3.1：**每车一小波，无时损可融，冲突时大波优先**。
- **门口预等**：每波 FIFO 派单后若有 **≤2 辆空闲车**，应并行前往门口等待区 **hub=(6,1)** 及其邻格（如 `(5,1)`、`(7,1)`），在当波车辆取货/送货时不阻塞走廊；取货完成后若等待车被挤离门口，送货前再拉回等待区。
- **取货分波**：到站即装（`pickup-stagger`），禁止先到取货台空等慢车整波同步装货。
- **送货/离场**：冠军路径（~446–455s）为 **via-leave**（送货+离场一体）+ 末波 **final-chain≤2**；取货用 joint sync load（非 stagger）。门口 hub-parallel / 分波 overlap 会破坏 via 可达性，默认关闭。

---

## 6. 轨迹输出

- CSV 列：`timestamp,name,X,Y,pitch,loaded,destination,Emergency,task-id`  
- **dense fill**：每个 AGV 每个 sim_t 一行，无缺口。  
- 全体 AGV 同步 timestamp（同 t 8 行）。

---

## 7. 校验（`validate_ok` 当且仅当全过）

- `n_collisions = 0`  
- `n_swaps = 0`  
- `n_hard_wall = 0`  
- `n_illegal_motion = 0`（含 teleport、同步非 ±1 步）  
- 无 premature_unload  
- **`n_pickup_cell_violations = 0`**：`loaded` 由 FALSE→TRUE 时，AGV 必须在该任务取货站的**交互格**（站台四邻 leave 环；站台格本身常为障碍，可含站台格）。禁止远处/错站取货。

**自检**：统计「同 (X,Y) 不同 pitch 的相邻行」→ 应有大量 turn 行；**不应**出现「移动行同时改变 pitch 且无前置 turn 行」。

---

## 8. MP4 渲染

| 输入 | 路径模式 |
|------|----------|
| 地图 | `map_editor/exports/SH_custom_{slot:02d}_position.csv` |
| 任务 | `..._task.csv` |
| 障碍 | `..._obstacles.json` → 读 **`extra_obstacles`** 数组（非裸 list） |
| 轨迹 | `results/.../SH_custom_{slot}_conflict_horizon_k{max_active}.csv` |

```bash
python simulation/display.py 1 --record out.mp4 \
  --task ..._task.csv --position ..._position.csv \
  --traj ...traj.csv --obstacles ..._obstacles.json --duration 120
```

障碍坐标 1-indexed；与 solver 同一 `extra_obstacles`。

---

## 9. 禁止事项（常见 AI 错误）

- ❌ 把 wave 当 sim_t（强制 1 wave = 1s 逐 tick 贪心）  
- ❌ 移动时 instant 改 pitch（跳过 turn 1s）  
- ❌ idle 车 `traj=[当前格]` 永不移动  
- ❌ 障碍 JSON 当数组解析（漏 `extra_obstacles` 键）  
- ❌ 冲突解算不考虑 turn 展开（会撞车）  
- ❌ 派单永远最近车（远端车永不接活）

---

## 10. 运行与成功条件

```bash
python -m ml_research.benchmarks.coord_custom_ai.solve_conflict_horizon \
  --slot 3 --max-active 4 --deadline 20000
```

成功：`completion_ratio ≥ 0.999` 且 `validate_ok: true`。

---

## 11. 安全优化清单（改前必读）

> **总则**：任何改动不得违反 §2–§7；不得触碰 §9 禁止项。  
> **晋升门槛**：SH03 `--max-tasks 20` 上 **validate_ok** 且 sim_t ≤ 基线×1.05 且全员 idle% 不升高，才允许合并；100 任务全量作为第二门槛。

**基线（当前）**：20 任务 · sim_t=820 · validate_ok · 单车 idle 40–68%

### O1 停车禁忌（anti ping-pong）— P0

| 项 | 内容 |
|----|------|
| **改什么** | `_pick_resume_park` / yield flee：维护 `recent_parks[agv]`（deque，K=6）；候选 park 在禁忌集则跳过 |
| **规则** | ✅ §4 仍选 remA* 更小（仅从剩余候选）· ✅ §3 让位 · 不改优先级序 |
| **验收** | 热点格如 `(2,6)` visit **降 ≥30%**；validate_ok；sim_t 增 **≤5%** |
| **回滚** | collision>0 或 completion<100% |

### O2 让位后强制 resume — P0

| 项 | 内容 |
|----|------|
| **改什么** | active 且 remA*>0：若连续 **wait≥3 tick**，下一 wave 必须 greedy/resume-step（仍过冲突解算） |
| **规则** | ✅ §4 resume 哲学 · ✅ §2 turn 展开 · 不让 keeper 跳过 T* |
| **验收** | active 车连续 wait>15 tick 的段数 **减少**；validate_ok |
| **回滚** | n_collisions>0 |

### O3 Turn-aware solo 规划 — P0

| 项 | 内容 |
|----|------|
| **改什么** | Phase A：turn-cost BFS（状态 `(x,y,pitch)`，同向 move=1、turn+move=2）→ xy 路径 → `_expand_full_timeline` |
| **规则** | ✅ §2 与 `navigation.py` · ✅ §3 solo+overlay · T* 仍在展开后时间线 |
| **验收** | carry 段 p95 **下降**；validate_ok；sim_t 可略增 **≤8%** |
| **回滚** | validate 失败或 sim_t 增 >10% |

### O4 Idle staging 分级 — P1

| 项 | 内容 |
|----|------|
| **改什么** | `_idle_staging_goal`：仅 **assign_count 最低且 manh≤D** 的 ≤max_active 辆 idle 向 pickup staging；其余 → side park（x∈{1,20} 或低 degree） |
| **规则** | ✅ §3 idle 禁止整局静止 · ✅ §9 公平派单 |
| **验收** | 无任务 idle tick **降 ≥20%**；主通道 visit **降**；validate_ok |
| **回滚** | 20 任务内某车 tasks=0 |

### O5 热点 pair 交替 — P1

| 项 | 内容 |
|----|------|
| **改什么** | chronic pair：yielder flee 禁止占 keeper 下一展开格；可选 token 交替 keeper |
| **规则** | ✅ §4 chronic · ✅ §3 swap/vertex |
| **验收** | 日志 stuck pair **降 ≥40%**；validate_ok |
| **回滚** | 新 swap 事件 |

### O6 批量提交 cap — P2（可选）

| 项 | 内容 |
|----|------|
| **改什么** | 无冲突时 commit 上限 `MAX_COMMIT=20`（展开后 sim 步） |
| **规则** | ✅ wave≠sim_t |
| **验收** | sim_t 降或持平；validate_ok |
| **回滚** | wall>2× 且无 sim_t 收益 |

### O7 派单拥堵感知 — P2（可选）

| 项 | 内容 |
|----|------|
| **改什么** | 派单 `(assign_count, manh+λ×corridor_load, name)`；load=过 pickup 邻格的 active 路径数 |
| **规则** | ✅ assign_count 仍第一关键字 · ✅ §9 |
| **验收** | 同取货口扎堆 **减少**；completion 100% |
| **回滚** | 任一车 tasks < floor(20/8)-1 |

---

## 12. 实施顺序与回归

```
O1+O2+O3 → 20任务回归 → O4+O5 → 20任务回归 → 100任务 → MP4
```

```bash
python -m ml_research.benchmarks.coord_custom_ai.solve_conflict_horizon \
  --slot 3 --max-active 4 --max-tasks 20 --deadline 5000
python ml_research/results/coord_custom_ai/conflict_horizon/_analyze_traj.py
```

**硬门禁**：validate_ok · tasks 全完成 · §9 无复现。  
**软门禁（至少一项）**：sim_t↓ · idle%↓ · 热点 visit↓ · stuck 日志↓。

---

## 13. 明确不做

| 方案 | 原因 |
|------|------|
| 1 wave=1s 逐 tick 贪心 | §9；3000s 仅 3/20 |
| idle 全员冲向最近 pickup | 占窄巷；idle 63%+ |
| 冲突不解 turn 展开 | §9；撞车 |
| 去掉 assign_count | §9；远端车永不接活 |
| 为 sim_t 牺牲 validate | §7 硬门禁 |
