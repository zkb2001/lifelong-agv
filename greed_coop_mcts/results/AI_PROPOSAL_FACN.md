# 调研备忘 + 创新提案：FACN-Net

调研时长：≥10 min（多源检索 + arXiv/LoRR/CIE 摘要精读 + 对照本栈 `fleet_cap` / SH20 几何）  
日期：2026-10-11  

**FACN** = **F**leet **A**ctivation & **C**oncurrency **N**etwork  
一句话：学一个**全局并发活跃度** \(K^*=\)「同时允许 busy 的 AGV 数」，在开阔/简单场景把闲置车拉进任务流，在拥堵场景主动限流——替代当前几乎静态的 TBC `max_fleet_busy∈{5,6,7}`。

---

## 1. 本栈现状（为何 SH20 是大金矿）

| 机制 | 作用 | 缺口 |
|------|------|------|
| `throat_band_fleet_cap` (TBC) | 开机定 `base_fleet_busy` | 几何分段粗；**几乎不向上开** |
| `_claim`：`busy >= fleet_cap` 停派 | 硬并发上限 | SH20 有 30 车却 cap≈7 |
| `busy_pulse` | 无进展时**降** cap | 只降不升；且 `n<12` 不脉冲 |
| StageNet / dest-gate | 姿态 / 谁先进廊 | **不管有多少车该干活** |

实测几何（无 map-id）：

| Slot | n | free | near_pad_deg2 | TBC | 理论闲置池 |
|------|---|------|---------------|-----|------------|
| SH01 | 8 | 363 | 14 | 6 | ~2 |
| SH11 | 16 | 216 | 83 | 5 | ~11 |
| **SH20** | **30** | **378** | **12** | **7** | **~23** |

SH20：开阔地板 + 大车队，TBC 只给 7 → **大半车队永久闲置**。用户判断正确：比 CAPRI 式「抢廊排序」更大的杠杆是 **把简单场景的活跃度打开**。

---

## 2. 文献地图（与「活跃车辆数」直接相关）

### A. 最优车队规模 / 主动减活（拥堵侧）
1. **LoRR’23 冠军复盘**（*Scaling LMAPF…*）：高密图上 **disable agents** 可大幅提吞吐；Game 4000 最优约 2500/4000；明确 future work = **search optimal agent number**（全局 + **局部区域**）。  
2. **LoRR’25 No-Man-Sky**：每步保留距目标最近的 \(n\) 车活跃，其余原地低优先级让路（Game \(n=2500\)）。  
3. **HARR / MS-MAPD-BHS**：service time 对 fleet size **非单调**——加车先降延迟，过密再升。  

### B. 运营级 fleet sizing（开阔/日常侧，与 SH20 同向）
4. **CIE’26 RMFS fleet sizing MLP**：把 fleet size 当成**反复运营决策**；仿真扫最优 \(K^*\)，MLP 预测；相对全开部署 **少用 10.6 车且 makespan −1.4%**。  
5. **Offline RL warehouse SLAM throughput control**：固定控制间隔改吞吐设定，history-informed state，惩罚下游拥堵——**节流/放流**与我们的 \(K\) 同构。  
6. **ESC 自适应调参**（FC digout，2026）：无模型、以吞吐为反馈调规划参数；适合做 FACN 上线后的慢自适应。  

### C. 分布 / 区域容量（比标量 \(K\) 更细，作 v1.5）
7. **GRAND**（2025）：GNN-RL 输出 **free agent 区域分布** → min-cost flow 再平衡 → 局部指派；LoRR 吞吐 +~10%。  
8. **Zone-capacity / Capacity-MAPF**：区/点容量约束；本栈 `region_admit` 已是弱版本。  
9. **STEAM**：瓶颈处 **时间延迟** + 密度 logit；不学 \(K\)，但说明「该停谁」可与「允许多少」解耦。  

### D. 对照（不宜整换）
10. **RHCR**：要的是持续 engage；开阔图 idle 是浪费。  
11. **CAPRI / 优先级学习**：管序不管量——正交，可后挂。  
12. **RuleNet / SoftGuidance**：方向场；本问题是 **claim 阀门**。  

### 调研结论
- 文献共识：**活跃规模有最优区间，且随布局/负载变化**；拥堵要减活，开阔要放活。  
- 本竞赛车队人数固定 → 学的不是「买几台车」，而是 **运营并发 \(K\)**（谁可 claim/保持 busy）。  
- 与 CAPRI 对比：CAPRI=微观准入序；**FACN=宏观阀门**——SH20 类收益应更大。  

---

## 3. 方案：FACN-Net

### 3.1 控制对象（输出）

**主头（v0，必做）**  
\[
K_t \in \{2,3,\ldots,\min(N, K_{\max})\}
\]
写入现有  
`fleet_cap = effective_fleet_cap(base_fleet_busy, …)`  
即替换/修正 `base_fleet_busy`，直接卡 `_claim` 的 `busy >= fleet_cap`。

推荐参数化（更稳）：  
\[
K_t = \mathrm{clip}\big(K_{\mathrm{TBC}} + \Delta_t,\; K_{\min},\; K_{\max}\big),\quad
\Delta_t \in \{-2,-1,0,+1,+2,+3,+4\}
\]
- \(K_{\mathrm{TBC}}\): 现 `throat_band_fleet_cap`  
- 开阔可 **+**，拥堵可 **−**（覆盖 LoRR disable 思想，但不物理删车）

**副头（v1，可选）**  
- `max_assigned`（同时接近 pickup 数）相对 \(K\) 的比例  
- 或 4–9 区 `region_cap` 缩放（GRAND-lite）

**明确不输出**  
- 不改 BFS；不 per-agent 方向场；不 Approach-Lease。

### 3.2 输入特征（无 map-id）

**静态 \(g\)（episode 常量）**  
- \(N\), `free`, `near_pad_deg2`, `#pickups`, `#dropoffs`  
- 密度 \(N/\mathrm{free}\)，空闲池比 \((N-K_{\mathrm{TBC}})/N\)  
- 分区 free 直方图（现成 `count_free_by_region`）

**动态 \(x_t\)（滑动窗，如 64 tick）**  
- `busy`, `idle`, `remaining`, `done_rate`（Δdone/窗）  
- `stall`, `avg_committed_H`, `n_repair/窗`, `conflict_heat`  
- `avg_region_occ`, `n_dest_gates`, 同 dest 载货簇大小  
- 近 pad deg≤2 占用人数（喉道负载）

全部归一化到 \([0,1]\)；**禁止 slot-id / 地图哈希**。

### 3.3 触发时机（何时前向）

| 时机 | 周期 | 理由 |
|------|------|------|
| **冷启动** | \(t=0\) | 先用静态 \(g\) 给 \(K_0\)（SH20 立刻抬） |
| **主周期** | 每 `MEET_H`（12）或每 24 tick | 与 MRSC 对齐，开销极低 |
| **事件触发** | `no_prog` / `done_rate` 骤降 / `busy` 触顶连续 ≥T | 该升却升不动，或该降 |
| **粘滞** | 决策后锁定 `TTL∈[24,48]`；\(\lvert\Delta\rvert\le 1\)/步 | 防 CAPRI 式 thrash |

**不在每 tick 的局部 PIBT 里调 \(K\)**——只动 claim 阀门。

### 3.4 网络与训练

**结构（小）**  
- MLP：`[g ⊕ x] → 64 → 64 → logits over Δ`（或对 \(K\) 的 分类）  
- 可选：1 层对 region token 的注意力后再池化（为 v1 区域头留接口）

**标签（推荐 IL，对齐 CIE’26）**  
1. **离线扫参 Oracle**：每图对 \(K\in\{4..12\}\)（大车队再加 14/16）跑完整/半程 episode；  
   \(K^*=\arg\min\{\mathrm{sim}\mid \mathrm{VALID}\land\mathrm{done}=100\}\)（并列取更大 \(K\) 的吞吐）。  
2. 行为克隆：\((g,x_{\mathrm{early}})\mapsto \Delta^*=K^*-K_{\mathrm{TBC}}\)。  
3. 难例：SH20/13/12/11 + 课程图；**门禁图**标签约束 \(\lvert\Delta\rvert\le 1\) 或强制 \(\Delta=0\)。  

**可选 RL 精调**：reward \(= +\Delta\mathrm{done_rate} -\lambda\cdot\mathbb{1}[\mathrm{stall\uparrow}]\)；动作为慢变 \(\Delta\)（类 ESC / offline RL throttle）。

### 3.5 硬约束 / Adaptive FACN Authority（AFA）

与 ASA 同哲学，**先几何门再网络**：

```
raise 允许 ⟺ near_pad_deg2 ≤ OPEN_HI 且 free/N ≥ DENS_LO   # SH20-class
lower 允许 ⟺ stall 高 或 喉道占用高 或 no_prog              # LoRR-class
否则        ⟺ Δ=0（纯 TBC）
```

门禁：SH01/02/03/08 默认 \(\Delta\in\{-1,0,+1\}\) 且需 AFA；全量前先 SH20 扫参证明上限。

### 3.6 与现模块关系

```
TBC → FACN(Δ) → effective_fleet_cap → _claim / max_assigned
                     ↑
              MRSC meeting 周期读状态
StageNet / dest-gate / E4 不变（微观）
```

叙事：**StageNet=姿态，FACN=车队油门，dest-gate=喉道红绿灯**——一个问题一个网。

---

## 4. 落地顺序

1. **Oracle 扫 \(K\)**：SH20 优先（期望 sim 明显低于 883）；对照 SH01/03/08 确保不炸。  
2. 写日志：`(g,x,K,Δdone,stall)`。  
3. 训 FACN-v0（Δ 分类）；挂 `effective_fleet_cap` / `base_fleet_busy`。  
4. 门禁 + SH20/13；再开 SH11/12 的 **降活** 半边。  
5. （可选）区域头 / ESC 慢自适应。  

### 成功判据
- SH20：sim 相对当前 **明显下降**（主证据）。  
- SH01/02/03/08：仍 ≤15% vs FULL。  
- SH11/12：不因抬 \(K\) 变差；理想情况靠 **降 Δ** 略好或持平。  

### 明确不做
- 不物理删车；不 map-id；不恢复 Approach-Lease / SAME_DROP_SERIAL。  
- 不重做 CAPRI 作为主创新（可后挂）。  

---

## 5. 备选（次优）

| 方案 | 评价 |
|------|------|
| 纯手调把 SH20 TBC 提到 10/12 | 能涨，但无自适应，难讲创新 |
| GRAND 全套区域 flow | 重；可作 FACN v1.5 |
| 只学 disable（LoRR） | 与用户「简单场景激活更多」方向相反，只覆盖半边 |

---

## 6. 对外一句话

> **FACN**：在终身 MAPD 中学习运营级并发活跃度——开阔图把闲置 AGV 放进任务流，拥堵图主动限流；几何协议与 StageNet 仍负责微观协调。
