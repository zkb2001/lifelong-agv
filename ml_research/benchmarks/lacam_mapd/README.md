# LaCAM*-MAPD（对比基线）

Lifelong MAPD on SH_custom_01..20（官方 100 单）。

## 两条路径

### A) 独立 lifelong（`solve_lacam`）
PIBT 推进 + 周期性 / 窗口化 LaCAM*；`busy_cap=3` 控制并发。

```bash
python -m ml_research.benchmarks.lacam_mapd.solve_lacam --slot 1
python -m ml_research.benchmarks.lacam_mapd.batch_100 --start 1 --end 20
```

结果：`ml_research/results/coord_custom_ai/lacam_mapd_100/`

### B) ECBS RHCR 外环 + LaCAM* 联合核（完备外环）
```bash
# joint_core=lacam 走 windowed LaCAM* 替代 WCBS
set AGV_JOINT_CORE=lacam
# 或 solve_ecbs(..., joint_core="lacam", legacy_wave_ecbs=True)
```

## 相对 windowed ECBS（compare_100）
- ECBS：20/20 done=100，avg wall≈543s，avg sim≈7364
- 独立 LaCAM*：简单图（如 SH01/02/05）done=100 且 wall 远低于 ECBS；窄通道硬图仍可能卡死（与纯 PIBT 同类）
- RHCR+LaCAM*：外环完备，但当前 LaCAM 核在硬图上成功率仍低于 WCBS，需继续加强搜索/预算

核心实现：`lacam_core.py`（LaCAM*）、`windowed.py`（RHCR）、`solve_lacam.py`（lifelong）。
