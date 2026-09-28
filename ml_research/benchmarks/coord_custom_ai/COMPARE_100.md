# Compare 100 — 五方法对照（官方 100 单）

统一场景：`SH_custom_01`–`20` 官方 task/position CSV，同一套 `validate_hybrid_trajectory`。

| ID | 方法 | 角色 |
|----|------|------|
| `pibt` | 反应式 PIBT lifelong | 无预留弱基线 |
| `m0` | 引擎时空 A*（无 ECBS） | 有 A\*、无冲突树 |
| `pp` | 优先规划 + 时空 A* | 有优先级预留、无 ECBS |
| `ecbs` | 窗口联合 ECBS | 有冲突树、无分层交接 |
| `hier` | M0 + 结构/stall 交接 + ECBS | 分层冠军 |

## 运行

```bash
# 冒烟：SH01 × 五方法
python -m ml_research.benchmarks.coord_custom_ai.compare_100 --start 1 --end 1

# 复用已有 pibt_mapd_100 结果
python -m ml_research.benchmarks.coord_custom_ai.compare_100 --start 1 --end 20 --reuse-pibt

# 只跑部分方法
python -m ml_research.benchmarks.coord_custom_ai.compare_100 --methods pibt,hier --start 1 --end 20 --reuse-pibt
```

输出：`ml_research/results/coord_custom_ai/compare_100/`  
（`results.jsonl`、`SUMMARY.md`、`trajectories/`）
