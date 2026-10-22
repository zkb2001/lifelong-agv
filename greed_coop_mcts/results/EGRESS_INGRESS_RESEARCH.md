# 窄廊 Egress↔Ingress 对顶：文献调研与可落地方案

**问题定义（已由 SH17/SH12 轨迹证实）**  
卸完空载仍停在 deg≤2 接近廊（或死胡同）内，未卸载货从外侧迎面进入；无侧湾 → 对顶 livelock。  
SH03 不触发：卸货点 `pad_deg≥3`、有侧出口。  
**约束**：不重开 `ENABLE_SAME_DROP_SERIAL`；不做 Approach-Lease 廊上 FIFO；无 map-id；门禁 SH01/02/03/08。

---

## 1. 文献地图（按与本问题的贴合度）

| 来源 | 核心机制 | 与本问题关系 | 对本栈可行性 |
|------|----------|--------------|--------------|
| **Matsui 2025 DE-aisle Swap** ([PDF](https://www.scitepress.org/Papers/2025/132438/132438.pdf)) | 死胡同廊内 blocker=Target、要进者=Initiator；双方退到路口，在双连通主区用 PIBT 换位；撤退期优先级反转；完成后禁止再入该廊；必要时 One-Push | **最贴合** SH12 Hangzhou 死胡同 + SH17 单行接近廊对顶 | 高：可做成状态触发的临时 swap 任务，挂 gap/PGFA |
| **Yamauchi et al. 2022 PIBTTP** ([arXiv:2205.12504](https://arxiv.org/abs/2205.12504)) | 树/死胡同内目标者获 **temporary priority**；他车必须退回主区；禁止进无关树；树内只走最短路；PIBTTP-TA 可侧枝暂避 | 给「廊内要出去的空载」临时最高优先，逼迎面载货退出 | 高：改优先级 + dest-gate，不必整换 PIBT |
| **SHARP / Fixed-Haven** ([arXiv HTML 2608.26759](https://arxiv.org/html/2608.26759); ScitePress SHARP) | 卸货后必须 **retreat 到专属 Haven**；busy/retreating 全程预留到 Haven 的路径；idle 只在 Haven；实证：固定回家是树状图鲁棒性关键 | 解释「卸完停在 task endpoint」有害；本栈有 home 但缺 **带优先的强制外撤** | 中高：做 SHARP-lite（无 SIPP 全预留），卸后 sticky egress |
| **Ma et al. 2017 Well-formed + Dummy path** ([arXiv:1705.10868](https://arxiv.org/abs/1705.10868)); MDPI 2024 endpoints | 只能在非任务 endpoint 久停；卸后加 dummy path 撤到空闲 EP | 同 Haven 思想；RHCR 文指出 hold/dummy 会损吞吐 | 中：已有 home；勿做成永久占 endpoint |
| **工业 VNA / lane lock** (Omega 2021 VNA；Multi-AMR lane lock 2026；WSC22 AGV) | 窄巷 **容量 1**；进巷前保证出场路径；**先出后进**；巷锁直到进出完成 | 直接对应「先出后进」业务规则 | 高：对 pad 的 deg≤2 前缀做软容量/方向锁 |
| **Corridor-CBS / MAC** (Li et al. corridor reasoning；Ni&Ye AAAI’26 MAC) | 同廊对向冲突一次按方向组解；多车同廊一组约束 | 思想：同廊同时只允许一个方向 | 中：本栈非 CBS；可软化成方向互斥 |
| **MACGA** (IJCAI 2025) | 建廊并 **疏散** 廊上他车；+PIBT | 像 PGFA keep_clear，但可为「外撤廊」服务 | 中：扩展焦点到 egress 受益人 |
| **RHCR Progress Potential** (Li et al. AAAI’21) | 窗内势不降则加窗 | 已有 prog_widen；对顶时加窗不够，需换协议 | 已部分落地；单靠它不够 |
| **WinC-MAPF** (AAAI’25) | 窗式完备：启发式更新防 livelock | 偏搜求解器 | 低：架构不匹配 |
| **Approach Lease / 硬串行同 dest** | 廊上 FIFO staging | 已在 SH02/03 否决 | **不做** |

---

## 2. 问题—机制对照

```
SH17: Bumblebee(空载, 廊深处) ↔ Sideswipe(载货, 迎面)
        │                              │
        ▼                              ▼
   需要 EGRESS                    需要 INGRESS
        │                              │
        └──── 单行 deg≤2，无侧湾 ─────┘
```

文献共识三句话：

1. **拓扑**：PIBT 完备性依赖双连通；死胡同/树必须额外协议（PIBTTP / DE-swap / Haven）。  
2. **角色**：卸完不得以 task endpoint 为休息点；必须 retreat（SHARP / dummy path）。  
3. **资源**：窄巷当容量-1 资源；有出者时禁止再进（VNA / lane lock / DE no-reenter）。

当前栈缺口：PGFA **永远偏 carrier 进 pad**，没有「廊内空载 egress」角色与临时最高优先 → 正是 SH17 sticky 2980 tick 的原因。

---

## 3. 推荐落地方案：CEBI（Corridor Egress-Before-Ingress）

在现有 `MRSC → dest-gate → gap → PGFA → prog` 上加一层 **状态触发** 协议（无 map-id）。

### 3.1 触发（几何，非地图名）

同时满足：

- 存在空载 `e`，所在格 `deg≤2`，且该格属于某 unload pad 的 BFS 接近前缀（或死胡同组件）；  
- 存在载货 `c`，其到同一 pad 的最短路穿过 `e` 的位置（或与 `e` 在同一 deg≤2 组件内对顶）。

→ 开启该前缀的 **Egress Session**，TTL 与 no_progress 共用计数。

### 3.2 动作（按文献优先级组合）

| 步骤 | 行为 | 文献来源 |
|------|------|----------|
| E1 | `e` 获 **temporary max priority**（高于一切 carrier），目标强制最近 `deg≥3`（非 home 也可，先出廊） | PIBTTP temp priority；SHARP retreat |
| E2 | 同前缀上所有 **ingress 向** carrier：`stage_goal` 改写到前缀外最近 deg≥3；**禁止**再踏入 session 前缀 | DE no-reenter；VNA lane lock；PIBTTP A4 |
| E3 | 若 carrier 夹在 `e` 与路口之间：对该 blocker 做 **向外推**（增大到 pad 的 Manh / 沿树退到根），深度≤3；必要时整段 sticky 对顶升级为 **swap retreat**（双方退到路口再放行） | DE-aisle Target retreat；已有 outward 可复用 |
| E4 | 卸货瞬间起 `T_egress` tick：该车标记 `retreating`，即使尚未检测到对顶也优先外撤离开前缀 | SHARP busy→retreating |
| E5 | Session 结束条件：`e` 到达 deg≥3，或前缀内无空载，或 TTL；之后恢复普通 dest-gate / PGFA | DE swap complete |

### 3.3 明确不做什么

- 不在远处改 FINAL / 不做廊上 FIFO holder 排队（Approach Lease）。  
- 不 demote carrier 任务身份，只改写阶段目标与禁入前缀。  
- 不上全量 SIPP Haven 预留（SHARP 全量）；本栈用 sticky 目标 + 优先级即可。  
- 不打开同 dest 硬串行全局开关。

### 3.4 与现有 PGCP 关系

| 已有 | 作用 | 对 egress 对顶 |
|------|------|----------------|
| Contested-Dest Gate | 拦「外面的人」进窄区 | **挡不住已在廊内的空载** |
| Outward Push | 挡路者外推 | 有用，但优先权仍偏 carrier |
| Progress widen / busy pulse | 加窗、降并行 | SH17 已大量触发仍不够 |
| Idle hygiene | 空载离廊 | 优先低，被 carrier 顶回 → 两格震荡 |

CEBI = 把「廊内空载」从 hygiene 提升为 **临时焦点受益人**，并给 ingress **硬禁入**。

---

## 4. 实施顺序（建议）

1. **E4+E1（SHARP-lite / PIBTTP-lite）** — 卸后 sticky egress + 临时最高优先；风险低，门禁友好。  
2. **E2（lane lock / no-reenter）** — session 期间禁 ingress；对准 SH17。  
3. **E3（DE-swap lite）** — 仅当 sticky 对顶 ≥N tick 时升级撤退换位；对准 SH12 死胡同。  
4. 每步过 SH01/02/03/08；再测 SH10/11/12/17。

验收信号：

- `sticky_idle↔loaded` 时长下降；  
- SH17 末段不再出现「空载钉在 `(12,9/10)` + 多车 Xiamen」；  
- `n_park_freezes` 相对 `n_corridor_clears` 上升（清廊能粘住）；  
- SH03 sim_time 波动 <15%。

---

## 5. 主要参考文献（可引用）

1. Matsui, T. *Improvement of PIBT-based Solution Method for Lifelong MAPD… DE aisles*. ICAART 2025.  
2. Yamauchi et al. *Deadlock-Free MAPD using Priority Inheritance with Temporary Priority*. arXiv:2205.12504, 2022.  
3. Hirayama & Yoshida (et al.). *Fixed-Haven Reservation… / SHARP*. 2026.  
4. Ma et al. *Lifelong Multi-Agent Path Finding for Online Pickup and Delivery*. AAMAS 2017.  
5. Li et al. *Lifelong MAPF in Large-Scale Warehouses (RHCR)*. AAAI 2021.  
6. Ni & Ye. *Multi-Agent Corridor Reasoning (MAC)*. AAAI 2026.  
7. Pertzovskiy et al. *MACGA*. IJCAI 2025.  
8. Okumura et al. *PIBT*. AIJ 2022.  
9. Luna & Bekris. *Push and Swap*. IJCAI 2011.  
10. Very Narrow Aisle AGV scheduling / lane-lock warehouse literature (Omega 2021; multi-AMR lane lock 2026).
