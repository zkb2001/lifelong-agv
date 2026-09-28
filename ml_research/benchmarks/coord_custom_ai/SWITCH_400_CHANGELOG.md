# SH01–20 ×400 开关栈改动日志

目标：**sim_t ≤ 8000** + **VALID**。

---

## 当前结论（必读）

1. **r110 = SH02 VALID**：sim=**33170**（优于 r89=41315 / r101=35334），ratio=0.995。
2. r110 瓶颈：`wall_first_ecbs`@t=75/done=9 打断健康 M0（~8–12 sim/task）→ ECBS ~80–100；末期 avg_wave≈1.5。
3. **r111**：延后/门控 wall_first；claim 抬到 min_k；禁 corridor/delivery jam 落到 k=1；目标压 sim≤8000。

## 已落地（相对 r89）

1. **撞取货台**：`blocked_plan = walls ∪ stations`；A*/ST 只允许 goal 踏上站台；ECBS/ST 回退均做 pad-transit 过滤；repair `station_pads` 禁穿行。
2. **并行疏散**：`_joint_wave_with_evac` — 批次失败时 1 车去停车点，与本波 pickup 同一次 `_prioritized_st_paths`；claim-cooldown 同样优先 joint-evac。
3. **pickup rising**：仅精确 pad 提交 + `rise_ok={pad}`；repair 仅当 `cand==target` 才允许 False→True。

## 运行表

| run | valid | sim_t | 备注 |
|-----|-------|-------|------|
| r89 | VALID | 41315 | 改前最佳 + 视频 |
| **r90** | INVALID | **36173** | pickup_cell=3；功能改动基线 |
| r91–r99 | INVALID | ~35–39k | rising/invent 调参 |
| r100 | INVALID | 37753 | pickup_cell=56；禁 invent 仍 phantom |
| **r101** | INVALID | **35334** | pickup_cell=**7**；premature=3；ratio=0.995 |
| r102 | INVALID | 37550 | 过激 strip/scrub → collision=3569 |
| r103 | INVALID | 37060 | 清 loaded → collision=3883 |
| r104 | INVALID | 36332 | 延迟弹队列仍差；collision=654 |
| r105 | INVALID | 37516 | held≈597k；station_pads 进 repair |
| **r106** | INVALID | **37337** | held=6064；pickup_cell=9 |
| **r107** | INVALID | **36401** | pickup_cell=4；whitelist 有效 |
| r108 | INVALID | 36165 | 仅 unload_dwell=1 |
| r109 | INVALID | 39119 | dwell 改 XY → collision |
| **r110** | **VALID** | **33170** | scrub+延迟卸货 dwell；比 r89/r101 更快 |
| r111 | abort | — | M0 门控生效但 A*=0.30→~3s/tick，墙钟风险 |
| r112 | abort | t=375 done=30 | sim/task≈12.5（外推≈5k）；墙钟 2s/tick，7200s 跑不完 |
| r113 | running→stop | t≈800 done≈49 | 长 M0；墙钟跑不完 |
| r114 | … | … | 卸货点13只停上/右，西侧主干让出，再进 ECBS 分波 |

## 日志

`batch_400_switch_20/run_sh02_r89.log` … `run_sh02_r101.log`  
`videos/SH02_r89_VALID_8000s.mp4`
