"""Custom-map solvers: ECBS + conflict-horizon (best-effort) + validation."""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "coord_custom_ai"
LOG_DIR = RESULTS_DIR / "logs"


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = ["PKG_DIR", "RESULTS_DIR", "LOG_DIR", "ensure_dirs"]
