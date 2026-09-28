# Hierarchical MAPD: Easy = True M0+SwapNet, Hard Transition Protocol

> **OUTDATED (2026-09):** Default solver path is now **M0 + SwapNet + TrafficNet
> recovery** (`park_artery_sem_v1`). SceneDifficulty → ECBS handoff is disabled
> unless `legacy_wave_ecbs=1`. See `ml_research/benchmarks/lane_traffic_ai/README.md`.

This document is the *legacy* architecture contract for lifelong mode switching in MioVerse.
Code entry points: `solve_ecbs.py` (`_solve_via_m0_engine`), `hierarchical_gate.py`
(`LifelongGateTracker`), `simulation/engine.py` (main copy / M0).

## Hard rule: Easy ⇒ true M0 + SwapNet

When the scene classifier commits **`easy`**:

- Run the continuous **main-copy** engine (`simulation/engine.py`) with optional **SwapNet**.
- **Never** use the wave-batched `"astar"` branch inside `solve_ecbs` (that path is
  multi-agent prioritized ST in lifelong waves — it is *not* M0).

Symptoms of the wrong path (historical bug): idle AGVs at opening, many pickups in
one synchronized wave, logs like `planner=astar wave=6` without `engine M0`.

`allow_mode_switch` must **not** divert an easy verdict away from M0. It only
enables mid-run escalation *out of* M0 when the committed band leaves easy.

## Literature (transition design)

| Idea | Work | Use here |
|------|------|----------|
| Resolve only a short window; grow horizon if stuck | RHCR (Li et al., AAAI 2021) | medium→hard deepen \(w\)/\(k\) |
| Adaptive horizon + reuse search state | ACCBS | warm-start constraints / paths across windows |
| Priority warm-start across queries | exRHCR / exPBS | sticky affinity + `seed_reserved` |
| Congestion guide before full joint search | Guide Path + PIBT (AAAI 2025) | soft pressure inside M0 (pipeline / SwapNet) before handoff |
| Escalate only uncertain subset | PRIMAL / PRIMAL3 | hybrid core = conflict subset, not full fleet |

Consensus: lifelong mode change is **continuous deepen + warm-start**, not
killing M0 and jumping to full-horizon ECBS.

## Three-band protocol

```text
easy  → 100% M0 engine + SwapNet
  │     (commit at most +1 band per decision)
medium → handoff window + small joint core + bystander A* (seeded)
  │     (deepen w / k, keep paradigm)
hard  → larger window / larger core (Prioritized default; ECBS fallback)
```

### Commit / hysteresis ([`commit_scene_label`](../coord_custom_ai/hierarchical_gate.py))

- Upgrade at most **one** band per decision: `easy → medium → hard`.
- A raw “sudden hard” while committed easy **must** land on **medium** first.
- Downgrade only after a calm streak (hysteresis). When A* conditions hold,
  **jump straight** `hard|medium → easy` (no medium ladder on the way down).
- Hotspot on easy must **not** invent medium by itself.
- `AllowDeescalate` still requires: past `handback_dwell` since last escalate,
  no recent joint_fail / no_progress / serial-recover, and not `force_hard`.
- Upgrade remains at most **+1** band: `easy → medium → hard`.

### Warm M0 handback (wave → true easy)

After sustained easy calm on the wave-astar **buffer**, switch demos
(`m0_handback=1`) pose-warm resume true M0+SwapNet:

```text
[HIER] M0 handback WARM t=... carriers=N -> engine M0+SwapNet
[M0] resume_from_snapshot t0=... swapnet=1 allow_yield=1
```

Cold CSV restart is forbidden (historical INVALID). Carriers are allowed;
handoff/handback snapshots are field-symmetric. Anti-pingpong: dwell after
escalate, ≤3 handbacks/episode, temporary ×1.5 re-escalate thresholds after handback.

Wave-astar after de-escalate is **not** the easy terminal state.

### Soft pre-escalation (still inside M0)

Before yielding, M0 already applies congestion tools (endpoint pipeline, SwapNet,
holding). Many “false hard” spikes should be absorbed without handoff.

### Runtime hard escalate (idle / surface / assign-fail)

SceneDifficulty alone may stay **easy** while the fleet is livelocked at a hub
(all AGVs idle, `surface_tasks` non-empty, A* thrashing). With
`allow_mode_switch=1`, M0 yield polls every `gate_every` ticks and **force-commits
at least medium** when any of:

| Signal | Default threshold |
|--------|-------------------|
| All idle (no business `task_id`) + surface/queues non-empty | streak ≥ 2 samples |
| Estimated `done` plateau while work remains | streak ≥ 3 samples |
| `sim._assign_fail_streak` (assign attempts fail with idle+surface) | ≥ 16 |

Force path: `LifelongGateTracker.force_commit_at_least("medium")` → still
`easy→medium` only (+1). Before writing the handoff snapshot, call
`evacuate_hub_idles_far(min_r=4, max_r=8)` so pad-blocking idles relocate away
from unload rings.

Inside M0, `hold_excess_near_congested_hubs` relocates idle on/near pads
(manh≤2) to holding at radius 4–8; assign-fail streak ≥ 8 shortens
`astar_wall_budget_s` and triggers the same far evacuate.

## M0 → medium handoff window

When `allow_mode_switch=1` and M0 is running:

1. Every `gate_every` ticks, call `LifelongGateTracker.decide(...)` with live pose / queues.
2. If committed label becomes `medium` or `hard`:
   - Freeze engine state (time, poses, loaded, bindings, remaining tasks).
   - Write partial trajectory prefix.
   - Export remaining task CSV + AGV snapshot.
   - Enter the lifelong **wave hybrid** loop at `now = sim.time`.
3. Wave loop uses residual paths / idle seeding (`seed_reserved`) where applicable
   (exRHCR-style warm start).
4. Final trajectory = M0 prefix concatenated with wave suffix; then validate as usual.

**Out of scope (this revision):** dropping back from wave loop to M0 when the band
calms to easy (avoids engine ping-pong). Documented as a follow-up.

## Medium → hard (deepen, do not rebrand)

| Knob | Medium | Hard |
|------|--------|------|
| Joint core size \(k\) | Small (`suggested_k` / `wave_hard_cap`) | Larger |
| Conflict horizon | Short plan/exec window | Longer / escalate |
| Joint solver | **Prioritized** spacetime (default) | Same; **ECBS** only if core fails or `AGV_JOINT_CORE=ecbs` |
| Bystanders | Prioritized ST with core reservation seed | Same |

## Joint-core policy (this stack)

- **Default:** `joint_core=prioritized` (fast, good for small cores).
- **Fallback:** ECBS retained behind the switch — quality / stubborn conflicts.
- **Later (not this revision):** PIBT / LaCAM as a fast third option when the core
  grows or times out; do **not** replace easy M0 with PIBT.

## Code map

| Piece | Role |
|-------|------|
| `_solve_via_m0_engine` | True easy path; optional yield; **warm_snapshot** handback resume |
| `_build_wave_handback_snapshot` / `_apply_m0_warm_snapshot` | Wave→M0 pose-warm |
| `LifelongGateTracker.decide` | Band + planner; dwell / deescalate veto |
| `run_sim_loop(..., should_stop=)` | Early stop for gate escalate |
| Wave hybrid (`_split_hybrid_movers`, `_plan_hybrid_wave_paths`) | Medium/hard after handoff |
| `DEFAULT_JOINT_CORE` | `prioritized` unless env/arg overrides |

## Logging expectations

Easy open (even with `--allow-mode-switch`):

```text
[HIER] scene=easy ... → still engine M0+SwapNet   # if allow_mode_switch
[HIER] scene=easy -> engine M0+SwapNet
[M0] start ...
```

Handback success:

```text
[HIER] M0 handback WARM t=... carriers=... -> engine M0+SwapNet
[M0] resume_from_snapshot t0=... swapnet=1 allow_yield=1
```

Must **not** open with:

```text
[HIER] allow_mode_switch=1 → skip M0 short-circuit
[ECBS] SH slot=...   # as first planner entry while still easy
```

Must **not** treat as success:

```text
calm recovered on wave-astar (skip cold M0 handback)
```