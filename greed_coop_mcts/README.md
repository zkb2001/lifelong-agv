# greed_coop_mcts

独立 lifelong MAPD。**主策略 RRSC**：全局学路权 → 约束阶段契约 → 单车按优先级栈贪婪推进。

## 设计（RRSC）

- 全局 AI：整图方向场 + guidance（边代价 / stay 惩罚），与具体车解耦。
- 阶段契约：`issue_contracts` 用加权路径生成 waypoints；与高优先级对头时降级为 YIELD。
- 局部：PIBT 式优先级栈，在不挡更高优先级前提下贪婪推进当前阶段。
- **无**窄巷检测、**无**硬编码任务波次；开图与密图同一套逻辑。

实现：`rule_obs.py` / `rule_net.py` / `rule_system.py` / `local_agent.py` / `train_rule_system.py`。

## 训练

```powershell
python -m greed_coop_mcts.train_rule_system --steps 5000 --batch 16 --device cpu
```

权重：`checkpoints/global_rule_transformer_v1.pt`

## 求解

```powershell
python -m greed_coop_mcts --slot 3 --policy rrsc --device cpu
```

- `--policy auto` / `rrsc`：RRSC-v2（路权→**ASA** 自适应 StageNet/teacher→区域软准入→优先级贪婪）
- `--no-stage-net`：强制 teacher；`--force-stage-net`：强制 StageNet（绕过 ASA）
- 默认栈还含 **TBC**（喉道带容量）与 **DLO**（双字典序 legalize）；无 map-id
- SH01–20（`PYTHONHASHSEED=0`）：avg **1712** vs FULL 1877（**−8.8%**），劣图 0，13/20 严格优于；详见 `results/CGPN_TERN_STATUS.md`
- `--policy glocal`：旧 GLocal-RH；`rules` / `xfmr` / `mcts` 备选

训练阶段选取：

```powershell
python -m greed_coop_mcts.train_stage_net --steps 4000 --device cpu
```
