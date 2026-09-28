# MioVerse AGV — 精简速查

> 保留：**Baseline（时空 A* + 贪心）**、**ECBS**、**RL-RH-PP**、**最大努力 conflict-horizon**，以及场景编辑 / 生成 / 校验 / 视频工具。

## 目录

```
competition/navigation.py          # 竞赛提交基线
simulation/{engine,display}.py     # 研究仿真 + 轨迹回放/录视频
data/                              # 标准 CSV
ml_research/
  common/                          # paths, load_main_copy
  map_editor/                      # SH_custom 场景编辑器 + exports/
  tools/render_video.py            # 轨迹 → MP4
  benchmarks/
    common.py / allocators.py / runner.py / verify_fifo.py
    scenarios/generator.py         # 压力场景生成
    rl_rh_pp/                      # Transformer 优先级 + RH-PP
    coord_custom_ai/
      solve_ecbs.py                # 窗口化 ECBS
      solve_conflict_horizon.py    # 最大努力原则求解器
      validate_hybrid_trajectory.py
      scenes.py
```

## 常用命令

```bash
# Baseline / PP
python -m ml_research.benchmarks.runner M0
python -m ml_research.benchmarks.runner M2

# ECBS（SH_custom_03）
python -m ml_research.benchmarks.coord_custom_ai.solve_ecbs --slot 3

# 最大努力 conflict-horizon
python -m ml_research.benchmarks.coord_custom_ai.solve_conflict_horizon --slot 3

# RL-RH-PP 训练+对照
python -m ml_research.benchmarks.rl_rh_pp.run_train_eval

# 场景编辑器
python -m ml_research.map_editor

# 压力场景生成
python -m ml_research.benchmarks.scenarios.generator

# 轨迹校验（ECBS/horizon 内已自动调用）
# validate_hybrid_trajectory / verify_fifo

# 录视频
python -m ml_research.tools.render_video --traj path/to.csv --out out.mp4
```
