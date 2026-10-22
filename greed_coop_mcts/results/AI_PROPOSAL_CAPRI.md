# 调研备忘 + 创新提案：CAPRI-Net

调研时长：≥10 min（多源检索 + 精读摘要/PDF + 对照本栈缺口）  
日期：2026-10-11  

---

## 1. 本栈现状（提案约束）

| 已有 | 角色 | 问题 |
|------|------|------|
| MRSC 开会 | 周期协调主干 | 优先级/准入多为启发式 |
| StageNet (ASA) | 选 FINAL/APPROACH/CLEAR | 选择性有用；不管「谁先进窄廊」 |
| RuleNet | 已默认关闭 | 消融无效 |
| PGCP dest-gate / TBC / E4 | 几何规则 | **同 dest 持有者 = 最近 Manh**，偏短视；flat 图难再挖 |
| SoftGuidance | 曾试 | directed_bfs 不吃 soft → 无效 |
| Approach Lease / SAME_DROP_SERIAL | 已否决 | 不可再用 |

**真正缺口**：不是再学方向场，而是学会 **在容量-1 喉道上「谁先进 / 谁等待 / 何时降并行」**——文献与未完成图诊断一致（同 dest 串车、egress↔ingress）。

---

## 2. 文献地图（≥15 条主线）

### A. 学习优先级（与开会最贴）
1. **LEAP-PBS**（2025）：监督预测 pairwise 优先级 → 初始化 PBS 根图；仓储存吞吐不掉、规划时间大降。  
2. **S2AN**（AAMAS’24）：注意力学 blocking 关系 → 无环优先级；限时大图优于纯 PIBT/PBS。  
3. **JAIR Learning-guided Prioritized Planning**（仓库 LMAPF）：RL 提全局优先序，挂滚动窗 PP。  

### B. 拥堵预测 / 软改路径代价
4. **ConvLSTM 拥堵预测**（大车队物流）：历史拥堵+计划路径 → 时空延迟场，改规划代价，吞吐 +4% 级。  
5. **STEAM**（2026）：无训练框架，从 cost-to-go 滚出热点 → 空间绕行 / 时间延迟 / 密度修正。  
6. **Congestion-Aware MAPD**（DNN 近似拥堵消息）：吞吐可翻倍（特定设定）。  
7. **窄廊 congestion metric + 单向预约**（arXiv:2202.11334）。  

### C. 任务/目标分配学习
8. **Attention for MAPD Task Allocation**（SAC’24）：注意力+RL 分任务，抑贪心冲突。  
9. **GNN Auction Q-learning**（NeurIPS’22）：多智能体指派近最优保证。  

### D. 交通图 / 单向边（对照已死 RuleNet）
10. **Online GGO / MGGO**（2024–2026）：优化边权甚至边方向；软引导为主。  
   → 与「再训一个方向场」同族，本栈 SoftGuidance/RuleNet 已证明难吃进 BFS。  

### E. 端到端模仿 / 搜索混合（对照，不宜整换架构）
11. **SILLM**（10k robots IL + SSC 通信）。  
12. **MAPF-GPT**（模仿 LaCAM，十亿轨迹）。  
13. **SCRIMP / PRIMAL2**（通信 RL）。  
14. **MATS-LP + CostTracer**（本仓库已有 ckpt；当前 `n_mcts=0`）。  
15. **Real-Time LaCAM + CS-PIBT**（学策略当 PIBT 启发）。  
16. **MAC / MACGA / PIBTTP / DE-aisle / SHARP**：协议思想已部分落入 CEBI/E4，缺学习决策头。  

### 调研结论（取舍）
- **不要**再做一个全图方向场 / SoftGuidance 2.0（MGGO/RuleNet 同坑）。  
- **不要**整栈换成 MAPF-GPT/SILLM（丢掉 MRSC 协议与竞赛校验契约）。  
- **要**做：在 **meeting 决策点** 上，把 LEAP/S2AN 式优先级学习 + 拥堵特征，对准本栈特有的 **contested unload session**。

---

## 3. 提案名称：**CAPRI-Net**  
**C**ongestion-**A**ware **P**riority **R**anking for **I**ngress

### 一句话
在 MRSC 开会时，对「竞争同一卸货喉道的车」做注意力排序，输出 **准入序 + 可选并行度修正**；几何协议（PGCP/E4）仍做硬约束，AI 只替换其中的「谁先」短视启发式。

### 为何比再挂一个杂模块更「深」
| 嵌入点 | 现逻辑 | CAPRI 替换/增强 |
|--------|--------|-----------------|
| `rewrite_contested_destinations` | sticky = 最近 Manh | **learned ranking**（可非最近） |
| `meet_pri` / PIBT 序 | StageNet 标量 + 手调 | CAPRI 分数注入 seed_pri |
| TBC / busy pulse | 固定 5/6/7 + 脉冲 | 头 2：**Δcap ∈ {-2,-1,0}**（仅当拥堵置信高） |
| CEBI/E4 | 几何触发 egress | 头 3（可选）：egress vs ingress 角色 logit |

StageNet **保留**（选阶段姿态）；CAPRI **专管资源竞争序**——二者正交，不再叠 RuleNet。

### 网络（小、可训）
**输入（每 meeting，每个 contested pad 一组）**  
对候选车 \(i\)（carriers + 近廊空载）：  
- 几何：Manh、BFS 前缀位置、`deg`、是否在 deg≤2、相对 pad 内外向  
- 局部占用：前缀上人数、对向计数、近邻热度  
- 进度：stall、载货时长、E4 retreating 标志  
- 可选全局：`near_pad_deg2`、fleet_cap、done 剩余  

**骨干**：小 Transformer / 双向注意力（S2AN/SCRIMP 风格），候选=token。  

**输出**  
1. listwise / pairwise 准入分数 → 排序  
2. （可选）`Δfleet_cap`  
3. （可选）每车 egress_boost  

**硬约束（保门禁）**  
- 仍禁止 Approach Lease；等待点必须 deg≥3  
- ranking 只在 **已触发 contested / CEBI session** 时启用，开阔图零开销  
- 无 map-id  

### 训练
1. **行为克隆**：从长时域/高 repair 成功轨迹标「最终先卸者」与「未导致 >T sticky 对顶的序」；  
2. 或 **离线 RL**：reward = −Δsim / +done，惩罚 sticky 邻接时长；  
3. 教师也可短窗用「随机序仿真前向 64 tick」挑吞吐最高序（可微弱搜索蒸馏，类 LEAP 用 PBS 标）。  

数据可从 SH01–40 + FULL 轨迹自动挖 contested 片段——比十亿 MAPF-GPT 轻两个数量级。

### 预期收益（假说，需实验）
- 打破 **flat**（SH02/09/13/14/17）：同 dest 序不再死锁于「永远最近者」；  
- SH11/12 类：learned 序 + Δcap 降灌入；  
- wall：仅 meeting×contested 前向，远低于 RuleNet 每 tick。  

### 明确不做
- 不恢复 RuleNet；不 SoftGuidance 软边权主路径  
- 不替换 BFS 最短路；不整换 LaCAM/MAPF-GPT  
- 不上廊上 FIFO staging  

---

## 4. 落地顺序
1. 日志：现有 dest-gate 触发时记录 (features, chosen_holder, outcome)。  
2. 离线挖片段 + 训 CAPRI-v0（仅 ranking 头）。  
3. 挂 `rewrite_contested_destinations`，门禁 SH01/02/03/08。  
4. 再开 Δcap 头；全套 SH01–40。  

### 对外叙事（防杂烩）
> **MRSC + 几何会话协议 + CAPRI**：在线会议式 MAPD 中，用注意力学习窄廊准入序。  
StageNet=阶段姿态；CAPRI=竞争资源排序——**一个问题一个网**。

---

## 5. 备选（次优）
| 方案 | 评价 |
|------|------|
| B. 在线 MGGO 边权/边向 | 与已失败 soft 引导同族 |
| C. 重开 CostTracer-MCTS 仅在 contested 簇 | 可作 CAPRI 的执行加强，不宜单独当主创新 |
| D. 任务分配 Attention（SAC’24） | 本竞赛任务流已定，增益点不在指派而在卸货竞争 |
