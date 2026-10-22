# MRSC Ablation (SH slots)

Protocol: leave-one-component-out under `policy=rrsc`.

| Variant | Stage | CGPN | TERN | PGCP | PGFA | full100 | avg done | avg sim(full) | avg wall | VALID |
|---------|-------|------|------|------|------|---------|----------|---------------|----------|-------|
| `full` | Y| Y| Y| Y| Y| 6/7| 94.6| 2126| 330.2s| 7/7 |
| `no_cgpn` | Y| —| Y| Y| Y| 5/5| 100.0| 2129| 144.8s| 5/5 |

## Per-slot: done / sim / wall(s)

| Slot | full | no_rulenet | no_stagenet | no_pgcp | no_pgfa | no_cgpn | no_tern | no_ai_guidance | no_ai |
|------|------|------|------|------|------|------|------|------|------|
| SH01 | 100/1207/37.13 | — | — | — | — | 100/1187/75.00 | — | — | — |
| SH03 | 100/3732/82.02 | — | — | — | — | 100/3186/93.08 | — | — | — |
| SH08 | 100/1431/51.06 | — | — | — | — | 100/972/110.38 | — | — | — |
| SH10 | 100/2499/53.01 | — | — | — | — | 100/1887/171.58 | — | — | — |
| SH11 | 100/3406/174.90 | — | — | — | — | 100/3411/274.15 | — | — | — |
| SH17 | 62/76291/1821.94* | — | — | — | — | — | — | — | — |
| SH20 | 100/484/91.53 | — | — | — | — | — | — | — | — |

Notes:
- `no_rulenet`: goal-flow teacher only (no GlobalRuleNet forward).
- `no_stagenet`: StageSelector teacher heuristics (no StageSelectorNet).
- `no_pgcp`: contested-destination gate + pad-egress off.
- `no_pgfa`: focal altruism off.
- `no_ai`: both nets off; PGCP/PGFA still on.
- `*` = incomplete (<100 tasks).
- CSV: `ablation_wall_sim.csv` (long), `ablation_sim.csv` / `ablation_wall.csv` / `ablation_done.csv` (wide).
