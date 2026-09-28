"""Smoke: ternary scene bands + sticky A* affinity in hybrid split."""
from __future__ import annotations

from ml_research.benchmarks.coord_custom_ai.hierarchical_gate import (
    LifelongGateTracker,
    band_scene_label,
    commit_scene_label,
)
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import (
    _planner_uses_hybrid_split,
    _split_hybrid_movers,
)


def test_band_scene_label() -> None:
    assert band_scene_label(0.05, t1=0.12, t2=0.35) == ("easy", "shrink_ok")
    assert band_scene_label(0.20, t1=0.12, t2=0.35) == ("medium", "shrink_ok")
    assert band_scene_label(0.50, t1=0.12, t2=0.35) == ("hard", "full_k")


def test_commit_scene_hysteresis() -> None:
    # easy→hard steps through medium (transition rung)
    lab, calm, down = commit_scene_label("easy", "hard", calm_streak=0, calm_needed=2)
    assert lab == "medium" and calm == 0 and down is False
    lab, calm, down = commit_scene_label("medium", "hard", calm_streak=0, calm_needed=2)
    assert lab == "hard" and calm == 0 and down is False
    # Downgrade needs calm, then jumps straight to raw (hard→easy, skip medium)
    lab, calm, down = commit_scene_label("hard", "easy", calm_streak=0, calm_needed=2)
    assert lab == "hard" and calm == 1 and down is False
    lab, calm, down = commit_scene_label("hard", "easy", calm_streak=1, calm_needed=2)
    assert lab == "easy" and calm == 0 and down is True
    lab, calm, down = commit_scene_label("medium", "easy", calm_streak=1, calm_needed=2)
    assert lab == "easy" and calm == 0 and down is True
    # Hotspot must NOT preempt easy → medium
    lab, _, _ = commit_scene_label(
        "easy", "easy", calm_streak=0, calm_needed=2, hotspot_pressure=True
    )
    assert lab == "easy"


def test_deescalate_dwell_blocks_hard_to_easy() -> None:
    gate = LifelongGateTracker(map_hardness=0.01, force=False, hard_cap=4, max_active=8)
    gate.deescalate_dwell_sim = 100
    gate.committed_scene = "hard"
    gate.scene_calm_streak = 1
    gate.last_escalate_sim_t = 50
    st = gate.decide(
        assigned={},
        queues={},
        recent_joint_fail=0,
        sim_t=80,  # 80 < 50+100 → veto
        no_progress_waves=0,
    )
    assert st.scene_label == "hard"
    assert "deescalate_veto" in st.reason
    st2 = gate.decide(
        assigned={},
        queues={},
        recent_joint_fail=0,
        sim_t=160,  # past dwell → jump hard→easy
        no_progress_waves=0,
    )
    assert st2.scene_label == "easy"
    assert st2.planner == "astar"
    assert "scene_downgrade:hard->easy" in st2.reason


def test_deescalate_veto_joint_fail() -> None:
    gate = LifelongGateTracker(map_hardness=0.01, force=False, hard_cap=4, max_active=8)
    gate.deescalate_dwell_sim = 0
    gate.committed_scene = "hard"
    gate.scene_calm_streak = 1
    gate.last_escalate_sim_t = -10**9
    st = gate.decide(
        assigned={},
        queues={},
        recent_joint_fail=1,
        sim_t=1000,
        no_progress_waves=0,
    )
    assert st.scene_label == "hard"
    assert "deescalate_veto:joint_fail" in st.reason


def test_gate_planner_ternary() -> None:
    gate = LifelongGateTracker(
        map_hardness=0.0,
        threshold=0.08,
        hard_cap=4,
        max_active=8,
        use_escalate_ai=False,
        use_map_obs=False,
        use_scene_diff=False,
        scene_t1=0.12,
        scene_t2=0.35,
        allow_mode_switch=True,
    )
    # Force raw bands via committed path without net: empty queues → easy
    st = gate.decide(assigned={}, queues={}, recent_joint_fail=0)
    assert st.planner == "astar"
    assert gate.last_scene_label == "easy"

    # joint_fail / hotspot on easy map stays A* (no premature medium)
    st = gate.decide(assigned={}, queues={"Rabbit": [{"task_id": "t1"}]}, recent_joint_fail=2)
    assert st.planner == "astar"
    assert gate.last_scene_label == "easy"

    # Degrade path: force_commit_at_least raises to medium (not bare hotspot)
    lab = gate.force_commit_at_least("medium", reason="unit_degrade")
    assert lab == "medium"
    st = gate.decide(assigned={}, queues={}, recent_joint_fail=0)
    assert st.planner == "hybrid"
    assert gate.last_scene_label == "medium"
    # Medium core should be hub-capable (floor 3), not stuck at 2
    assert int(st.suggested_k) >= 3

    # Climb toward hard via force_commit (SceneNet off: map hardness alone
    # must NOT invent medium/hard — see map_hardness_keep_easy).
    gate_h = LifelongGateTracker(
        map_hardness=0.5,
        threshold=0.08,
        hard_cap=4,
        max_active=8,
        use_escalate_ai=False,
        use_map_obs=False,
        use_scene_diff=False,
        allow_mode_switch=True,
        deescalate_dwell_sim=0,
    )
    st = gate_h.decide(assigned={}, queues={}, recent_joint_fail=0, sim_t=0)
    assert st.planner == "astar"
    assert gate_h.last_scene_label == "easy"
    assert gate_h.force_commit_at_least("medium", reason="unit", sim_t=10) == "medium"
    st = gate_h.decide(assigned={}, queues={}, recent_joint_fail=0, sim_t=10)
    assert st.planner == "hybrid"
    assert gate_h.last_scene_label == "medium"
    assert _planner_uses_hybrid_split(st.planner)
    assert gate_h.force_commit_at_least("hard", reason="unit", sim_t=20) == "hard"
    st = gate_h.decide(assigned={}, queues={}, recent_joint_fail=0, sim_t=20)
    assert st.planner == "ecbs"
    assert gate_h.last_scene_label == "hard"


def test_sticky_split_keeps_astar_affinity() -> None:
    assigned = {
        "A": {"destination": "Beijing", "pickup_point": (2, 2)},
        "B": {"destination": "Beijing", "pickup_point": (3, 3)},
        "C": {"destination": "Tianjin", "pickup_point": (4, 4)},
        "D": {"destination": "Shanghai", "pickup_point": (5, 5)},
    }
    pose = {
        "A": (1, 1, 90),
        "B": (1, 2, 90),
        "C": (1, 3, 90),
        "D": (1, 4, 90),
    }
    affinity = {"A": "astar", "B": "astar", "C": "joint", "D": "joint"}
    static = set()

    def bfs_len(a, b, _s):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    ecbs, astar, mode = _split_hybrid_movers(
        assigned,
        pose,
        k_budget=4,
        hard_cap=4,
        escalate=False,
        top_dest="Beijing",
        wave_net=None,
        map_hardness=0.2,
        map_obs=None,
        static=static,
        work_count={n: 0 for n in assigned},
        bfs_len=bfs_len,
        affinity=affinity,
        scene_label="hard",
    )
    assert "A" in astar and "B" in astar
    assert "A" not in ecbs and "B" not in ecbs
    assert "sticky" in mode

    # medium: hub-capable core (floor 3, ≤ hard_cap)
    ecbs_m, astar_m, mode_m = _split_hybrid_movers(
        assigned,
        pose,
        k_budget=8,
        hard_cap=4,
        escalate=False,
        top_dest="Beijing",
        wave_net=None,
        map_hardness=0.2,
        map_obs=None,
        static=static,
        work_count={n: 0 for n in assigned},
        bfs_len=bfs_len,
        affinity={"A": "joint", "B": "joint", "C": "joint", "D": "joint"},
        scene_label="medium",
    )
    assert 3 <= len(ecbs_m) <= 4
    assert len(ecbs_m) + len(astar_m) == 4


def test_run_sim_loop_should_stop() -> None:
    """Unit: should_stop forces early exit from run_sim_loop."""
    from ml_research.benchmarks.common import run_sim_loop

    class _FakeSim:
        def __init__(self) -> None:
            self.time = 0
            self.task_states = {"S": [1, 2, 3, 4, 5]}
            self.surface_tasks = {}

        def all_over(self) -> bool:
            return self.time >= 1000

        def time_forward(self) -> None:
            self.time += 1

        def steps_reorganize(self) -> dict:
            return {}

    sim = _FakeSim()
    steps, forced = run_sim_loop(
        sim,
        max_time=10_000,
        label="unit/should_stop",
        progress_every=10_000,
        should_stop=lambda s: int(s.time) >= 7,
    )
    assert forced is True
    assert int(sim.time) == 7
    assert isinstance(steps, dict)


def test_force_commit_at_least_medium() -> None:
    gate = LifelongGateTracker(
        map_hardness=0.0,
        threshold=0.08,
        hard_cap=4,
        max_active=8,
        use_escalate_ai=False,
        use_map_obs=False,
        use_scene_diff=False,
        allow_mode_switch=True,
    )
    assert (gate.committed_scene or "easy") in ("", "easy")
    lab = gate.force_commit_at_least("medium", reason="unit_idle_surface")
    assert lab == "medium"
    assert gate.last_scene_label == "medium"
    assert gate.escalated is True
    # Already at medium: no-op stay
    lab2 = gate.force_commit_at_least("medium", reason="again")
    assert lab2 == "medium"


def test_m0_stall_signals_idle_surface() -> None:
    from ml_research.benchmarks.coord_custom_ai.solve_ecbs import _m0_stall_signals

    class _Agv:
        def __init__(self, name: str, tid=None) -> None:
            self.name = name
            self.task_id = tid
            self.state = (1, 1, 0, 90)
            self.steps = []

    class _Sim:
        def __init__(self) -> None:
            self.time = 100
            self.agvs = [_Agv("A"), _Agv("B"), _Agv("C", "escape_C")]
            self.task_states = {"Ox": [{"task_id": "Ox-1"}]}
            self.surface_tasks = {"Ox-1": {"pickup_point": (2, 2)}}
            self._assign_fail_streak = 16

    sig = _m0_stall_signals(_Sim(), n_tasks=10)
    assert sig["idle_all"] is True
    assert sig["has_work"] is True
    assert sig["n_idle"] == 3
    assert sig["assign_fail_streak"] == 16
    assert sig["done_est"] >= 0


def main() -> int:
    test_band_scene_label()
    test_commit_scene_hysteresis()
    test_gate_planner_ternary()
    test_sticky_split_keeps_astar_affinity()
    test_run_sim_loop_should_stop()
    test_force_commit_at_least_medium()
    test_m0_stall_signals_idle_surface()
    print("[ok] medium sticky hybrid smoke passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
