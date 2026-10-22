# FACN P0 held-out (train SH01–20 only → test SH21–40)

Checkpoint: `d:\compitition\MioVerse\final_version\greed_coop_mcts\checkpoints\facn_p0_heldout.pt`

| Slot | FACN K | Δ | upd | FACN sim | best K | best sim | gap | tag |
|------|--------|---|-----|----------|--------|----------|-----|-----|
| SH21 | 6 | 0 | 0 | 1085 | 8 | 959 | +126 | MISS |
| SH22 | 6 | 0 | 0 | 1133 | 8 | 823 | +310 | MISS |
| SH23 | 6 | 0 | 0 | 1178 | 7 | 1142 | +36 | NEAR |
| SH24 | 8 | 2 | 14 | 875 | 15 | 811 | +64 | MISS |
| SH25 | 6 | 0 | 0 | 1626 | 10 | 988 | +638 | MISS |
| SH26 | 7 | 2 | 0 | 1522 | 8 | 1093 | +429 | MISS |
| SH27 | 7 | 1 | 0 | 1174 | 7 | 1174 | +0 | BEST |
| SH28 | 8 | 2 | 12 | 1198 | 11 | 1084 | +114 | MISS |
| SH29 | 6 | 0 | 0 | 2298 | 7 | 2296 | +2 | NEAR |
| SH30 | 8 | 2 | 22 | 1118 | 8 | 967 | +151 | BEST |
| SH31 | 5 | 0 | 84 | 4019 | 7 | 2582 | +1437 | MISS |
| SH32 | 6 | 0 | 0 | 1162 | 8 | 1053 | +109 | MISS |
| SH33 | 6 | 0 | 0 | 941 | 6 | 941 | +0 | BEST |
| SH34 | 6 | 0 | 0 | 1234 | 6 | 1234 | +0 | BEST |
| SH35 | 6 | 0 | 0 | 1141 | 8 | 827 | +314 | MISS |
| SH36 | 6 | 1 | 0 | 1170 | 8 | 958 | +212 | MISS |
| SH37 | 7 | 1 | 15 | 1502 | 6 | 1022 | +480 | MISS |
| SH38 | 5 | 0 | 0 | 1133 | 7 | 985 | +148 | MISS |
| SH39 | 6 | 0 | 27 | 1257 | 9 | 1056 | +201 | MISS |
| SH40 | 6 | 0 | 0 | 1298 | 10 | 1131 | +167 | MISS |

## Summary

- BEST: **4/20**
- NEAR: **2/20**
- MISS: **14/20**
- online updates>0: **6/20**

Compare to in-oracle v2 eval (`FACN_EVAL_K_21_40.md`): BEST 7 / NEAR 2 / MISS 11 — P0 measures true held-out drop.
