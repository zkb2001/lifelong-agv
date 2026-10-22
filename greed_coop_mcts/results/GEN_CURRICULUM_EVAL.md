# Generated maps SH21–40 vs current rrsc stack

- maps: 20 (easy=7, medium=7, hard=6)
- all free-space 4-connected: **yes** (asserted at gen)
- complete 100/100: **20/20**
- complete 100/100 VALID: **7/20**
- avg sim (VALID only): **1193.3**
- avg sim (done=100): **1600.3**

| Slot | diff | style | n | near | obs | sim | done | ok | mot | asa | cebi | e4 |
|------|------|-------|---|------|-----|-----|------|----|-----|----|------|----|
| SH21 | easy | open | 8 | 24 | 18 | 1008 | 100/100 | 1 | 0 | 0 | 0 | 0 |
| SH22 | easy | sprinkle_mild | 12 | 26 | 18 | 1252 | 100/100 | 1 | 0 | 1 | 0 | 0 |
| SH23 | easy | corridor_mild | 9 | 21 | 16 | 1246 | 100/100 | 0 | 23 | 0 | 0 | 0 |
| SH24 | easy | open | 15 | 33 | 18 | 1101 | 100/100 | 1 | 0 | 1 | 0 | 0 |
| SH33 | easy | sprinkle_mild | 10 | 25 | 18 | 900 | 100/100 | 0 | 4 | 1 | 0 | 0 |
| SH34 | easy | open | 6 | 26 | 18 | 1258 | 100/100 | 1 | 0 | 0 | 0 | 0 |
| SH35 | easy | hutong_mild | 10 | 47 | 43 | 961 | 100/100 | 1 | 0 | 1 | 0 | 0 |
| SH25 | medium | corridor_hard | 12 | 43 | 34 | 1523 | 100/100 | 0 | 230 | 1 | 0 | 0 |
| SH26 | medium | hutong | 12 | 60 | 52 | 1368 | 100/100 | 0 | 18 | 1 | 0 | 0 |
| SH27 | medium | sprinkle | 16 | 66 | 44 | 1431 | 100/100 | 1 | 0 | 1 | 0 | 0 |
| SH28 | medium | corridor_hard | 18 | 45 | 34 | 1497 | 100/100 | 0 | 73 | 0 | 0 | 0 |
| SH36 | medium | sprinkle | 14 | 70 | 55 | 1362 | 100/100 | 0 | 10 | 1 | 0 | 0 |
| SH37 | medium | hutong_mild | 16 | 59 | 45 | 1400 | 100/100 | 0 | 25 | 1 | 0 | 0 |
| SH38 | medium | maze_mild | 10 | 54 | 41 | 1446 | 100/100 | 0 | 4 | 1 | 0 | 0 |
| SH29 | hard | maze | 8 | 108 | 81 | 2133 | 100/100 | 0 | 42 | 1 | 0 | 0 |
| SH30 | hard | hutong | 20 | 55 | 49 | 1342 | 100/100 | 1 | 0 | 1 | 0 | 0 |
| SH31 | hard | maze_mild | 10 | 50 | 34 | 6591 | 100/100 | 0 | 4072 | 1 | 0 | 0 |
| SH32 | hard | hutong | 9 | 48 | 48 | 1463 | 100/100 | 0 | 4 | 0 | 0 | 0 |
| SH39 | hard | sprinkle_dense | 18 | 95 | 70 | 1245 | 100/100 | 0 | 20 | 0 | 0 | 0 |
| SH40 | hard | maze | 10 | 94 | 64 | 1479 | 100/100 | 0 | 10 | 1 | 0 | 0 |

## Failures

- SH23 [easy/corridor_mild] n=9 near=21: INVALID coll=0 swap=0 mot=23 fifo=0
- SH33 [easy/sprinkle_mild] n=10 near=25: INVALID coll=0 swap=0 mot=4 fifo=0
- SH25 [medium/corridor_hard] n=12 near=43: INVALID coll=0 swap=0 mot=230 fifo=0
- SH26 [medium/hutong] n=12 near=60: INVALID coll=0 swap=0 mot=18 fifo=0
- SH28 [medium/corridor_hard] n=18 near=45: INVALID coll=0 swap=0 mot=73 fifo=0
- SH36 [medium/sprinkle] n=14 near=70: INVALID coll=0 swap=0 mot=10 fifo=0
- SH37 [medium/hutong_mild] n=16 near=59: INVALID coll=0 swap=0 mot=25 fifo=0
- SH38 [medium/maze_mild] n=10 near=54: INVALID coll=0 swap=0 mot=4 fifo=0
- SH29 [hard/maze] n=8 near=108: INVALID coll=0 swap=0 mot=42 fifo=0
- SH31 [hard/maze_mild] n=10 near=50: INVALID coll=0 swap=0 mot=4072 fifo=0
- SH32 [hard/hutong] n=9 near=48: INVALID coll=0 swap=0 mot=4 fifo=0
- SH39 [hard/sprinkle_dense] n=18 near=95: INVALID coll=0 swap=0 mot=20 fifo=0
- SH40 [hard/maze] n=10 near=94: INVALID coll=0 swap=0 mot=10 fifo=0
