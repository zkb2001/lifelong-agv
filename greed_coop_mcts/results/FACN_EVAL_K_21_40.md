# FACN vs K-sweep — SH21–40

FACN one run (may adjust Δ online) vs fixed-`force_fleet_cap` sweep. **Caveat:** SH21–40 were in the v2 oracle used to train `facn_v2.pt` (reproduction check, not pure held-out).

| Slot | TBC | FACN K | Δ | updates | FACN sim | best K | best sim | FACN−best | rank |
|------|-----|--------|---|---------|----------|--------|----------|-----------|------|
| SH21 | 6 | 6 | 0 | 0 | 1085 | 8 | 959 | +126 | MISS/2 |
| SH22 | 6 | 8 | 2 | 0 | 864 | 8 | 823 | +41 | BEST/1 |
| SH23 | 6 | 7 | 1 | 0 | 1142 | 7 | 1142 | +0 | BEST/1 |
| SH24 | 6 | 10 | 4 | 0 | 602 | 15 | 811 | -209 | NEAR/6 |
| SH25 | 6 | 8 | 2 | 0 | 1269 | 10 | 988 | +281 | MISS/3 |
| SH26 | 5 | 7 | 2 | 0 | 1522 | 8 | 1093 | +429 | MISS/5 |
| SH27 | 6 | 7 | 1 | 0 | 1174 | 7 | 1174 | +0 | BEST/1 |
| SH28 | 6 | 7 | 1 | 14 | 1283 | 11 | 1084 | +199 | MISS/10 |
| SH29 | 6 | 8 | 2 | 0 | 2579 | 7 | 2296 | +283 | MISS/3 |
| SH30 | 6 | 8 | 2 | 0 | 981 | 8 | 967 | +14 | BEST/1 |
| SH31 | 5 | 5 | 0 | 0 | 3437 | 7 | 2582 | +855 | MISS/2 |
| SH32 | 6 | 8 | 2 | 0 | 1041 | 8 | 1053 | -12 | BEST/1 |
| SH33 | 6 | 8 | 2 | 0 | 791 | 6 | 941 | -150 | NEAR/2 |
| SH34 | 6 | 6 | 0 | 0 | 1234 | 6 | 1234 | +0 | BEST/1 |
| SH35 | 6 | 8 | 2 | 0 | 846 | 8 | 827 | +19 | BEST/1 |
| SH36 | 5 | 6 | 1 | 0 | 1170 | 8 | 958 | +212 | MISS/7 |
| SH37 | 6 | 8 | 2 | 0 | 1086 | 6 | 1022 | +64 | MISS/11 |
| SH38 | 5 | 5 | 0 | 4 | 1168 | 7 | 985 | +183 | MISS/5 |
| SH39 | 6 | 7 | 1 | 0 | 1236 | 9 | 1056 | +180 | MISS/9 |
| SH40 | 6 | 6 | 0 | 0 | 1298 | 10 | 1131 | +167 | MISS/4 |

## Summary

- FACN hits best K (or equal sim): **7/20**
- Near (≤50 sim or ≤5%): **2/20**
- Miss: **11/20**
- Slots with mid-run Δ change (`n_updates>0`): **2/20**

Online adjust: `FacnController.maybe_update` every `FACN_PERIOD` / TTL / stall events; `n_updates` counts Δ changes after cold start.
