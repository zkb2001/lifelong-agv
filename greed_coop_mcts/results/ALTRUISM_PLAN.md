# 全局利他系统优化方案（Progress-Gated Focal Altruism）

## 1. 当前利他机制清单

| 机制 | 位置 | 本意 | 实际副作用 |
|------|------|------|------------|
| idle corridor clear | `meeting_rh.apply_idle_corridor_clear` | 闲车离开 busy BFS，给载货让路 | remaining≤5 / stall 时常开；找不到离廊格仍每 tick 改 desire → 空转计数、贴廊钉死（SH03） |
| gap yield | `local_gap_yield` | 低优侧移给高优 | 高密度下互让无净进展（SH11）；无受益人焦点 |
| endgame unlock BFS | `solve.py` stall∧remaining≤5 | 再清廊 + 忙车优先步进 | 与 clear / sticky 三重改写 desire，互相覆盖 |
| sticky leader | `solve.py` Last-5 | 独占开路权 | 方向对；但若 leader 选错且 clear 无效，仍冻 |
| Stage CLEAR 偏置 | `run_meeting` carrier_stuck | 闲车让空间 | 无进度门控 |
| approacher drop | stall pulse | 减并行 | 过频饿死取货；过稀不解载货堵 |

## 2. 为什么「利他」会卡死（根因）

1. **无条件利他**：触发看 stall/remaining，不看「这一让是否缩短了某辆关键车到目标的距离」。
2. **无焦点受益人**：同时为所有 busy 清廊 → idle 在多条走廊之间振荡，或钉在多走廊交点（SH03 Sideswipe@(7,8)）。
3. **对称互让**：A 让 B、B 让 C，无人继承「必须前进」的权利 → livelock（文献：PIBT 靠 priority inheritance；我们只有 yield）。
4. **多层利他互殴**：同一 tick 内 gap → clear → endgame unlock → sticky 连续改写，后写覆盖前写，策略噪声大。
5. **有效性缺失**：清廊失败仍算「做了利他」，系统以为在协作，其实几何未变。
6. **与吞吐冲突**：过早串行/过冷战（已关）会伤 SH02/SH08；说明利他必须是「局部损、全局进度升」，不是「禁止做事」。

## 3. 设计原则

1. **损己可以，但必须服务唯一焦点受益人**（focal beneficiary）。
2. **进度门控**：受益人的 BFS/Manh 近目标才算利他成功；否则升级（扩 exclusive / 冻旁车）或换 lead。
3. **角色不对称**：idle/approacher 可大力利他；carrier 只对更高焦点让路，不对同级互让。
4. **停车冻结**：闲车离廊成功后 stick park，禁止每 tick 重选目标。
5. **单通道改写**：每 tick 只走一条「全局利他」管道，取消与 sticky/clear 的打架。
6. **保 SH01/SH02**：无 map-if；焦点利他在 stall 低时几乎不触发；endgame 仍积极清廊但带进度门控。

## 4. 架构：Progress-Gated Focal Altruism (PGFA)

```
每 tick (MRSC):
  meeting / follow+gap
  → PGFA.apply(desired):
       1) select_beneficiary(carriers|assigned, pri, streak, sticky)
       2) busy_path = BFS_prefix(beneficiary only)   # 焦点走廊
       3) 非受益人若在 path/邻域 → 离廊一步（损己）
       4) 闲车离廊成功 → park_freeze[n]=TTL
       5) 受益人优先步进；旁车不得占用 path
       6) 记录 beneficiary_dist；若 stall 且 dist 不降 → escalate / switch
  → legalize
```

## 5. 落地优先级

| P | 项 | 状态 |
|---|----|------|
| P0 | `global_altruism.py` + 接入 solve，替换打架的 endgame 多重清廊 | 实施 |
| P0 | 焦点走廊 + park freeze | 实施 |
| P1 | gap yield：仅当 mover 为受益人或 pri 差大；carrier 不对非受益 carrier 让 | 实施 |
| P1 | 进度失败换 lead（沿用 sticky jam 逻辑） | 实施 |
| P2 | Adaptive MEET_W（已有，保留） | 保留 |
| — | drop 串行 | **保持 OFF** |

## 6. 回归门禁

- SH01 ≤1.3k VALID 100
- SH02 100 VALID（sim 可略升，勿再 96 卡终局）
- SH08 100 VALID
- SH03 探针：冲过 96 或明显压 stall 时长
