# Scene Difficulty（场景难度分类）

用 **地图 + 近地平线任务 + AGV 规模** 预测 easy/hard。

## 标签与特征（v3）

| 项 | 定义 |
|----|------|
| **标签** | 本波优先时空 A*（reservation ST）能否把第一波 AGV 送到取货点 |
| **近地平线** | 最多看 `max(n_agvs, 2·n_agvs)` 条待办任务（非整段 lifelong 积压） |
| **规划器分支** | easy → 时空 A*；hard → WCBS/ECBS |

旧版用整局 M0 `completion_ratio` 打标，和门控「选本波规划器」不对齐，且深队列冷启动易假 hard。

```bash
# 用已有 samples 重刷特征+波次标签（推荐，无需重跑整局 M0）
python -m ml_research.benchmarks.hier_coord.scene_difficulty.refresh_horizon --workers 4

python -m ml_research.benchmarks.hier_coord.scene_difficulty.train \
  --epochs 70 --batch 32 --hard-weight 1.0 --min-hard-recall 0 --seed 1

python -m ml_research.benchmarks.hier_coord.scene_difficulty.eval \
  --test-data ml_research/results/hier_coord/scene_difficulty_test
```

## 决策（平衡单阈值）

| `p_hard` | `is_hard` | `k_policy` |
|----------|-----------|------------|
| `≥ threshold` | hard | `full_k` |
| `< threshold` | easy | `shrink_ok` |

标定默认按 **balanced accuracy**。

## 规划器分支

| SceneDifficulty 标签 | 本波规划器 | WaveNet 分波 |
|----------------------|------------|--------------|
| **easy** | **优先时空 A\*** | 关闭 |
| **hard** | **联合 WCBS/ECBS** | 可 escalate + shrink |

## Lifelong 阶段压力测试

```bash
python -m ml_research.benchmarks.hier_coord.scene_difficulty.run_lifelong_phase_stress \
  --slots 1,3,5 --n-pre 24 --n-burst 24 --n-post 24 --tag smoke
```
