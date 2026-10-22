# ASA + FG-CEBI + Multi-Band TBC + E4（一类论文向）

## 结论（可复现）

`PYTHONHASHSEED=0` + `device=cpu` + 每图子进程隔离：

| 指标 | FULL（旧） | **当前** | Δ |
|------|------------|----------|---|
| 完成 | 20/20 VALID | **20/20 VALID** | — |
| **avg_sim** | 1877.2 | **1707.9** | **-169.2 (−9.0%)** |
| 劣于旧系统 | — | **0 / 20** | 全面不劣 |
| 严格优于 | — | **14 / 20** | — |
| GATE | — | **PASS** | ✓ |

### 严格增益 14 图

SH01−46, SH03−697, SH05−62, SH06−268, SH07−586, SH08−35 (FG-CEBI),
SH10−145, SH11−677 (Dense-TBC), **SH12−90 (E4 sticky)**，SH15−165 (Mid-TBC),
SH16−448, SH18−93, SH19−9 (FG-CEBI), SH20−64.

打平：SH02/04/09/13/14/17。

## 默认创新栈（无 map-id）

1. **ASA** — `(n, near)` → StageNet/teacher  
2. **FG-CEBI** — `n≥10 ∧ near≥100` egress 廊锁（只改目标）  
3. **Multi-Band TBC** — mid / dense / mega 容量带  
4. **E4 SHARP-lite** — `n≥19 ∧ near∈[70,85]` 卸后 sticky 撤出 deg≤2 喉道  
5. **DLO + Pad-egress + Contested-dest + Progress pulse**

## 本轮否决（会回归）

| 尝试 | 结果 |
|------|------|
| Contested×Egress tick 重排 / holder yield | SH11 +621 |
| 宽 CEBI / tiny CEBI (near≥110) | SH17 死锁 |
| Mid-TBC LO→45 / Dense 扩 n≤20 | SH02+241 / SH12+24 / SH11 退化 |
| 无门控 E4（全图 sticky） | SH02/08/11/15/17 崩 |
| SoftGuidance λ∈{0.10,0.35} 开放图 | 与 BASE 完全打平（motion BFS 不含 soft） |
| 稀疏 soft-pulse (n≤9∧near≥110) | SH17 卡在 done=84 死锁 |
| Open-band Mid-TBC / Mid-TBC LO45 | SH02+241 门禁风险 |
| SMPP/SCPO / APA / DPSM / CGPN@SH12 | 既有否决 |

## 命令

```powershell
$env:PYTHONHASHSEED='0'
python -m greed_coop_mcts.batch --slots 1-20 --policy rrsc --device cpu
python greed_coop_mcts/results/_cmp_v_full.py
```
