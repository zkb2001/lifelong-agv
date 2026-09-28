"""Stress-test scenario catalog for AGV allocation benchmarks."""
from .generator import SCENARIO_DIR, generate_all_scenarios, load_scenario_manifest

__all__ = ["SCENARIO_DIR", "generate_all_scenarios", "load_scenario_manifest"]
