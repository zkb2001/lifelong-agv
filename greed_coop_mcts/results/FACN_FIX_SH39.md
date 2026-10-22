# FACN 翻车根因与修复（SH39 / SH11）

日期：2026-10-11

## 根因

SH39（FACN 旧）：`n=18`, `near_pad_deg2=107`, `Δ=-2→K=4`, `avg_parallel_busy=2.04`,
`n_facn_updates=8016`, sim=81601 / 98/100。

死亡螺旋：

1. **错误挂载**：`n≥16` 即挂 FACN，即使走廊极密（near≫40）也不 raise-eligible。
2. **错误限流**：`afa_allow_lower` 仅凭 `no_prog` 就允许降并发；网络/teacher 推到 `Δ=-2`。
3. **吞吐塌陷**：claim 上限过低 → 长期 `no_prog` → 每 tick 重算 FACN（4 万次 neural）→ 锁死在低 K。
4. SH11 同模式：`near=83`, `Δ=-2→K=3`, sim 1707→2407。

## 修复

1. `afa_allow_attach`：raise-eligible **或** `(n≥16 ∧ near≤60)`；SH11/SH39 类洪水图回纯 TBC。
2. `afa_allow_lower`：必须 **结构拥堵**（throat/cluster）∧ **压力**（stall/no_prog）。
3. `Δ=-2` 仅极端 pileup；`current_k` 软地板 `max(4, K_TBC-1)`（仍有任务时）。
4. 吞吐恢复：连续低 `done_rate` 时把 `Δ` 往 0 爬。
5. 更新节流：`FACN_MIN_GAP=12`，禁止裸 `no_prog` 每 tick 唤醒。

## 复测（修复后 `--facn`）

| Slot | 修复前 FACN | 修复后 | 判定 |
|------|-------------|--------|------|
| SH39 | 81601 / 98/100 / on Δ=-2 | **1260 / 100/100 / off** | 回 TBC，翻车消除 |
| SH11 | 2407 / on Δ=-2 | **1707 / off** | 回 TBC |
| SH20 | 671 / on Δ=+4 | **671 / on Δ=+4** | 提并发收益保留 |
| SH13 | 741 / on Δ=+2 | **741 / on Δ=+2** | raise 保留 |
| SH28 | 1361 / on | **1497 / on** | 仍优于 noFACN 1548 |
| SH12 | 1722 / on | **1468 / off** | 回 TBC |
| SH01 | 919 / off | **919 / off** | GATE OK |
