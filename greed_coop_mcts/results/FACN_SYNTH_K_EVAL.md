# FACN synth P1 held-out eval

ckpt=`d:\compitition\MioVerse\final_version\greed_coop_mcts\checkpoints\facn_v3_k.pt` test_n=20

| Map | FACN K | best K | FACN sim | best sim | gap | tag |
|-----|--------|--------|----------|----------|-----|-----|
| SYN_12833 | 9 | 8 | 1264 | 1098 | +166 | MISS |
| SYN_10349 | 16 | 16 | 727 | 695 | +32 | BEST |
| SYN_11719 | 10 | 6 | 677 | 977 | -300 | BEST |
| SYN_12511 | 10 | 7 | 820 | 1023 | -203 | BEST |
| SYN_11810 | 10 | 6 | 665 | 774 | -109 | BEST |
| SYN_10540 | 10 | 10 | 1331 | 1085 | +246 | BEST |
| SYN_11890 | 7 | 7 | 1672 | 1672 | +0 | BEST |
| SYN_10361 | 5 | 5 | 2917 | 2917 | +0 | BEST |
| SYN_10081 | 9 | 9 | 1514 | 1153 | +361 | BEST |
| SYN_10221 | 10 | 7 | 856 | 1111 | -255 | BEST |
| SYN_12026 | 7 | 8 | 1047 | 932 | +115 | MISS |
| SYN_11800 | 9 | 8 | 1059 | 1208 | -149 | BEST |
| SYN_11596 | 6 | 6 | 1022 | 1022 | +0 | BEST |
| SYN_12723 | 7 | 8 | 1975 | 1823 | +152 | MISS |
| SYN_11887 | 7 | 9 | 1091 | 958 | +133 | MISS |
| SYN_11715 | 9 | 8 | 1270 | 1440 | -170 | BEST |
| SYN_12461 | 8 | 6 | 1151 | 1012 | +139 | MISS |
| SYN_10134 | 9 | 9 | 1016 | 979 | +37 | BEST |
| SYN_10636 | 9 | 8 | 1610 | 1683 | -73 | BEST |
| SYN_12082 | 16 | 8 | 501 | 771 | -270 | BEST |

- BEST **15** NEAR **0** MISS **5**
- BEST+NEAR rate **75.0%** gate≥60%: **PASS**
