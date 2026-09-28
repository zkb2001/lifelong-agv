# Park / Artery / Semaphore（规则恢复，默认无 TrafficNet）

失败恢复栈：全图默认 **M0 A\* + SwapNet**；路径/分配失败时用 **频率标定主干道/停车点 + 规则信号量**  
驱动「取货 → 近卸货停车 → 信号量放行」。

**不再**使用 SceneDifficulty 分类器或 ECBS 作为默认路径（`legacy_wave_ecbs=1` 可回退旧路径）。  
运行时**不依赖** ParkArterySemaphoreNet；该网络仅保留在 `train_bc` / 实验路径。

## 规则恢复

1. 离线：`freq_probe` 笛卡尔覆盖 → `artery` / `parking`（卸货格+四邻；取货**仅八邻**、不含取货格本身；障碍邻格跳过）
2. 安装稳定标签：`python -m ml_research.benchmarks.lane_traffic_ai.map_labels`
3. 运行时：`attach_traffic_recovery(sim, meta=meta)` → `RuleParkArteryPolicy`
4. 等待车：规则信号量（path_clear / pad_free / congestion / connectivity）
5. 同一卸货口每 tick 只放行距离最近的一台
6. **不改写**活跃车已承诺路径（活跃时空投影为硬障碍）

## 频率标定

```bash
python -m ml_research.benchmarks.lane_traffic_ai.freq_probe --slots 1,2,3 --n-aug 32
# 把最新 run 安装到 artery_maps/
python -m ml_research.benchmarks.lane_traffic_ai.map_labels
```

输出：`ml_research/results/lane_traffic_ai/freq_probe/run_*/`  
稳定标签：`ml_research/results/lane_traffic_ai/artery_maps/SH_custom_XX.npz`

## 训练（可选 / 遗留）

```bash
# 单元测试
python -m ml_research.benchmarks.lane_traffic_ai.selftest

# 可选：用频率样本训 BC（运行时默认不用）
python -m ml_research.benchmarks.lane_traffic_ai.train_bc --steps 400 \
  --freq ml_research/results/lane_traffic_ai/freq_probe/<run_dir>
```

## Legacy v4

`TrafficRuleNet` / `LaneTrafficPolicy` / `step_cost` 仍保留（车道边权塑形），主路径已切到规则恢复。
