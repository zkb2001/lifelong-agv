# Paper figures (compare_100)

Generated from `COMPARE_FULL.csv` + Hier trajectory paths.

| File | Content |
|------|---------|
| `fig1_aggregate_bars` | Full100 / VALID / avg sim / avg travel (4 panels) |
| `fig2_completion_heatmap` | Per-map completion %; `×` = INVALID |
| `fig3_hier_handoff` | Automatic M0-only vs runtime handoff→ECBS |
| `fig4_pareto_sim_travel` | ECBS vs Hier scatter (full-completion maps) |

## Hier path labels (from traj filename)
- **M0 only** (4): SH01, SH04, SH06, SH20
- **Handoff → ECBS** (16): SH02, SH03, SH05, SH07, SH08, SH09, SH10, SH11, SH12, SH13, SH14, SH15, SH16, SH17, SH18, SH19

Regen:
```bash
python -m ml_research.scripts.plot_compare_100_paper_figs
```
