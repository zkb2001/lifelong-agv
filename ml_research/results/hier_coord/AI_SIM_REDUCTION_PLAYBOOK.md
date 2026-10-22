# AI → 压低 sim / 解决 Hier 切换

## 文献 → 本仓库切换问题

| 论文 | 主张 | 对应我们的坑 |
|------|------|--------------|
| Skrynnik *When to Switch* | **Assistant**：默认快策略，失败才换 | Stall 全队硬切 ECBS |
| Dergachev 2021 locally confined | 只升级死锁子集，再 **切回** | 只有单向 handoff |
| Hybrid RL-MAPF 2025 | 子集 **T_lock** 防振荡 | 解卡下一拍又整队切 |
| LNS2+RL | 用近期质量自适应 | 只有墙钟/plateau |
| MATS-LP CostTracer | 局部神经 MCTS prior | idle 无 goal → Wait |

## 本轮优化（v3）

1. **训 CostTracer IL** 4000 step → `mats_local_cost_tracer_v1.pt`（acc≈0.80）
2. **local_opt**：idle 合成疏散 goal + CostTracer MCTS（n_sims=64）+ LNS fallback
3. **switcher**：T_lock **只**在 assign / ≥2 纯 MCTS 移动时启用（LNS force 不算）

## 检验（sim↓）

| map | base | 上轮冠军 | local_opt | switch_v3 | 判定 |
|-----|------|----------|-----------|-----------|------|
| **SH09** | 3366 | switcher **2471** | **2808** (−534 vs 上轮 local) | v3=3968 / v3b=4313 | local_opt 大胜上轮 local；仍次于旧 switcher |
| SH12 | 3932 | local **3816** | 3954 (+138) | 4480 回退 | CostTracer 栈略伤；保留上轮 local |
| SH02 | 4117 | local **3776** | 3847 (+71) | — | 压过 base，略逊上轮 local |

**判定**
- **CostTracer + disperse 对 SH09 有效**（local 3342→2808），对 SH12/SH02 略负
- switcher 本轮 mid_dead / 晚切 **伤 sim**；保留旧 SH09 switcher 冠军（2471）
- 推荐：默认 `AGV_LOCAL_MCTS=1`（上轮无 CostTracer 栈对 SH12/02 更稳）；SH09 可叠旧 switcher；CostTracer local_opt 作为 SH09 无 switcher 时的强备选

## 代码 / 结果

- `mats_local/hier_hook.py` — disperse goals + CostTracer
- `hier_coord/tier_switcher.py` — T_lock 不计 LNS force
- `eval_ai_opt_v3.py` → `AI_opt_v3_ab.json`
- ckpt: `ml_research/checkpoints/mats_local_cost_tracer_v1.pt`
