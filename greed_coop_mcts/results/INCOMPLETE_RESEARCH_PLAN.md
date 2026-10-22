# 未完成图调研与方案（MRSC+PGFA 基座上）

**约束**：不重开 `ENABLE_SAME_DROP_SERIAL`；不复用已否决的 Approach-Lease（廊上 FIFO staging）；门禁 SH01/02/03/08 不退化。

---

## 1. 现象（轨迹 + 报表，对照健康图）

| Slot | AGV | done | H_cmt | repair | adapt | clr | park | busy | 卡住形态（末 2k tick） |
|------|-----|------|-------|--------|-------|-----|------|------|------------------------|
| SH01 | 8 | 100 | 3.7 | 69 | 0 | 5 | 4 | 4.6 | 闲车在宽格休息；无廊上对顶 |
| SH08 | 12 | 100 | 3.8 | 612 | 0 | 48 | 24 | 5.0 | 偶发廊邻接，能消化 |
| SH10 | 8 | 82 | 3.0 | 8688 | 454 | 12k | 103 | 5.0 | **同廊 y=6**：Jazz/Smokescreen/Optimus 载货钉死；Jazz↔Smokescreen 邻接 1275 tick |
| SH11 | 16 | 35 | **1.2** | 12k | 7k | 16k | 144 | **6.0** | 多车同 dest=Xiamen 挤 y=6；大量 idle 钉在 home 行；H 塌缩 |
| SH12 | 20 | 50 | **1.5** | 10k | 7k | **30k** | **6** | **6.0** | 同 dest=Hangzhou；clr 狂推但 park≈0（打地鼠）；密度最高 |
| SH17 | 8 | 62 | 3.5 | 13k | **0** | 9k | 2 | 6.0 | 3–4 载货同 dest=Xiamen 锁死在 (12–13,*) 团簇；Hound↔Ironhide **整段 2001 tick 贴邻**；H 未塌故 adapt 永不触发 |

**共性（与地图特征）**  
- 走廊细胞占比不高（~2%），但 unload 前缀落在 **deg≤2 单廊** 上 → 局部 capacity-1。  
- 失败时 **多辆 TO_DROP 指向同一 destination**，在单廊对顶/串车。  
- VALID=碰撞层干净；失败是 **活性 / livelock**（repair 只保证无碰，不保证前进）。  
- SH11/12：车队规模大 + `max_fleet_busy=6` 顶满 + region_occ↑ + H 塌。  
- SH17：H 仍健康 → 现有 `adaptive_meet_w(stall∧H)` **漏检**「无任务进展」死锁。

---

## 2. 文献映射（与本栈对应）

| 文献要点 | 对本项目的含义 |
|----------|----------------|
| **RHCR**（Li et al. AAAI’21）：短窗 PP 吞吐好但不完备；用进度势函数 \(P(w)\) 加窗 | 我们已有加窗，但条件错：SH17 要的是 **done/载货势不降** 而非仅 H 塌 |
| **RH-PP**（JAIR’26）：window repair = 安全修复，**不保证 progress**；循环互等仍 VALID | 解释 repair 爆炸却 done 停滞 |
| **PIBT / PIBTTP**（Okumura；临时优先级）：挡路者继承优先级并 **沿树/廊向外推**；死胡同需 swap/one-push | PGFA 的 `_pick_evac` 对载货挡路者有时仍 **朝 pad**；缺「向外推」与有界 backtrack |
| **DE-aisle swap**（PIBT 扩展’25）：窄廊死胡同双方退到路口再交换 | SH10/17 的同廊对顶更适合 **有向推退到 deg≥3 路口**，而非廊上排队 |
| **TP / well-formed / SBDA / dummy path / SHARP Haven** | Idle **不得以 task endpoint 为永久休息点**；要有合法旁停/撤退。本栈已有 home + PGFA park，但 **同 dest 争夺** 未解决 |
| **Corridor-CBS / MAC** | 廊冲突应按「方向组」一次解，而非逐格互让 → 对应「同廊同向只放行一人近 pad」 |

**已否决**：廊上 FIFO staging（Approach Lease）——在 SH02/03 制造车队堵死 holder。

---

## 3. 代码层缺口（回归阅读）

1. `adaptive_meet_w`：只看 `stall∧avg_committed` → SH17 漏检。  
2. `apply_focal_altruism`：单焦点有用，但同 dest 多车时焦点轮换 + 清廊朝 pad，变成 **打地鼠**（SH12 park=6/clr=30k）。  
3. `region_admit`：保护全部 `TO_DROP`，**不限制**「同一 unload 前缀上的载货数」。  
4. `max_fleet_busy=6`：在 16–20 车地图上把窄区灌满（SH11/12 busy≈6）。  
5. `local_gap_yield`：无「沿廊向外」方向约束；高密度互让无净进展（与 ALTRUISM_PLAN 一致）。

---

## 4. 推荐方案：Progress-Gated Corridor Protocol（PGCP）

在 **现有 MRSC → gap → PGFA** 上加三层薄机制（无 map-id 分支；无 drop 硬串行）。

### A. Contested-Destination Gate（几何触发的同 dest 准入）

**触发**（同时满足）：  
- ≥2 carriers 的 unload cell 相同；且  
- 其中 ≥2 人满足 `Manh(pad)≤K` **或** 位于该 pad BFS 前缀的 deg≤2 格上。

**动作**：  
- sticky holder = 最近 pad 者（与 PGFA closest-first 对齐）；  
- 非 holder：**不 demote 角色/任务**，只把 `stage_goal` 改到 **最近 deg≥3 的路口/侧湾**（必须离廊，禁止落在前缀上）；  
- holder 无进展 `≥T` → 轮换（与 PGFA `no_progress` 共用计数）。

**与失败 Lease 的区别**：只在「已经挤进窄区」时门控；等待点强制 deg≥3；不在远处改写 FINAL。

### B. Outward Push Inheritance（PIBTTP-lite）

当焦点受益人的下一步被占，且 blocker 在 deg≤2：  
- blocker **临时继承** bene 优先级；  
- 合法移动优先：**增大相对 pad 的 Manh**（向外推），禁止踏入 bene 前缀；  
- 深度 ≤3 的链式推（有界 backtrack）；推不动则 bene 本 tick 等待，下一 meeting 强制加窗。

挂在 `local_gap_yield` / PGFA holder 驱逐处，**替换**「挡路载货也可 prefer_pad」的逻辑。

### C. Progress Potential（真 RHCR）

除 H 塌外，增加：

```
no_task_progress = (done 连续 ≥T_done 不变)
                 ∨ (Σ Manh(carrier→pad) 连续 ≥T_d 不降)
```

则：  
1. `MEET_W → MEET_W_ADAPT` 并立即 `should_meet`；  
2. **脉冲** `max_fleet_busy: 6→3`（TTL≈2×MEET_H），压注入窄区的并行度；  
3. PGFA 禁止换 lead（锁死当前 closest holder）直到 Manh 下降或卸货。

SH17 正是此条要救的；SH11/12 同时受益于降并行。

### D. 轻量 Idle 廊洁（非 Haven 大改）

Idle 若连续占据「任一载货 BFS 前缀 ∩ deg≤2」超过 N tick → 强制一步到最近 deg≥3；  
**禁止**用 escape 覆盖 carrying/assigned 目标（已验证有害）。

---

## 5. 数据流（每 tick）

```
goals
 → meeting: StageNet → region_admit
 → Contested-Destination Gate (A)     # 仅几何竞争时
 → plan_committed / repair
 → gap_yield + Outward Push (B)
 → PGFA (焦点清廊；挡路载货只外推)
 → Progress Potential 触发加窗/降 busy (C)
 → legalize
```

---

## 6. 效率与门禁

| 机制 | SH01/02/08 | SH10/17 | SH11/12 |
|------|------------|---------|---------|
| A 未触发 | 零开销 | 解同廊串车 | 降同 dest 灌入 |
| B 偶发推 | 极少 | 打开对顶 | 打开对顶 |
| C 脉冲降 busy | 少触发 | 救「H 健康无进展」 | 降密度 |

**验收**：SH01/02/03/08 仍 100 VALID，sim 波动 <15%；SH10/11/12/17 的 `tasks_completed` 单调上升；关注 `n_corridor_clears / n_park_freezes` 比值下降（清廊要「粘住」）。

**落地顺序**：C（漏检修复，风险低）→ B（外推，对准对顶）→ A（同 dest 门控）→ D；每步过门禁再动 incomplete 四图。

---

## 7. 明确不做

- Approach Lease / 廊上 FIFO staging  
- 打开 `ENABLE_SAME_DROP_SERIAL`  
- 全 carrier 并集 keep_clear  
- 按地图 id 特判  
- 无进度条件的盲目加窗（已证明在 H 健康时无效）
