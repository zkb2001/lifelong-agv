"""Paths for the standalone greed–coop MCTS package (repo-root sibling)."""
from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent

RESULTS = PKG_ROOT / "results"
CKPT = PKG_ROOT / "checkpoints"
TRAJ = RESULTS / "trajectories"
MAP_EXPORTS_SYNTH = PKG_ROOT / "map_exports_synth"

# Shared simulation assets (data only — not ECBS / Hier solvers).
ENGINE_PY = REPO_ROOT / "simulation" / "engine.py"
MAP_EXPORTS = REPO_ROOT / "ml_research" / "map_editor" / "exports"

COSTTRACER_CKPT = CKPT / "cost_tracer_v1.pt"
TRANSFORMER_CKPT = CKPT / "mapd_transformer_v1.pt"
RULE_CKPT = CKPT / "global_rule_transformer_v1.pt"

for d in (RESULTS, CKPT, TRAJ, MAP_EXPORTS_SYNTH):
    d.mkdir(parents=True, exist_ok=True)
