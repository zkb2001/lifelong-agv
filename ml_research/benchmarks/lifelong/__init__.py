"""Lifelong MAPD: online task generator + live AGV monitor for baseline runs."""

from .task_generator import LifelongTaskGenerator, install_lifelong_generator

__all__ = ["LifelongTaskGenerator", "install_lifelong_generator"]
