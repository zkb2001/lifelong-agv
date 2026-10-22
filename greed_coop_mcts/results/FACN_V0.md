# FACN-v0 落地记录

日期：2026-10-11

## 做了什么

1. `facn_net.py`：MLP 预测 \(\Delta\in\{-2..+4\}\)，AFA 几何门，粘滞 TTL
2. 挂点：`base_fleet_busy = K_TBC+Δ` → `_claim` / `effective_fleet_cap`
3. Oracle 扫参 + `train_facn` → `checkpoints/facn_v1.pt`
4. CLI：`--facn` / `--force-fleet-cap`

## Oracle（SH20）

| K | sim | done |
|---|-----|------|
| 7 (TBC) | 883 | 100 |
| 10 | **695** | 100 |
| 12 | **695** | 100 |
| 14 | 706 | 100 |

相对 TBC：**−188（−21%）**

## 烟测（紧 AFA：仅 raise-eligible 或 n≥16 挂 FACN）

| Slot | sim | 先验/FULL | FACN | 判定 |
|------|-----|-----------|------|------|
| SH01 | **919** | 919 / 965 | off | GATE OK |
| SH02 | **2021** | 2021 / 2021 | off | GATE OK |
| SH03 | **3743** | 3743 / 4440 | off | GATE OK |
| SH08 | **1285** | 1285 / 1320 | off | GATE OK |
| SH20 | **671** | 883 / 947 | on Δ=+4 → K=11 | **−212 (−24%)** |

GATE：**PASS**。主收益在开阔大车队 SH20。
