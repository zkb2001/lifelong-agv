"""Config for ParkArterySemaphoreNet (v5) + legacy TrafficRuleNet (v4) helpers."""
from __future__ import annotations

from pathlib import Path

from ml_research.common.paths import CKPT, RESULTS

PKG = "lane_traffic_ai"
RESULTS_DIR = RESULTS / PKG
LOG_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"

# Legacy v4 (dir/phase/zone edge shaping) — kept for optional cost_field demos.
CKPT_PATH_V4 = CKPT / "lane_traffic_rules_aligned.pt"
CKPT_TAG_V4 = "lane_traffic_rules_v4"
N_MAP_CHANNELS_V4 = 11

# v5: parking + artery + connectivity + semaphore
CKPT_PATH = CKPT / "park_artery_sem_v1.pt"
CKPT_TAG = "park_artery_sem_v1"
N_MAP_CHANNELS = 12
N_SEM_FEATURES = 8

GRID = 21
PLAY_LO = 1
PLAY_HI = 20

N_DIRS = 4
DIR_DELTA = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIR_NAMES = ("E", "N", "W", "S")

N_ZONES = 4
BLOCK = 4

W_AGAINST_DIR = 2.5
W_RED_PHASE = 4.0
W_CROSS_ZONE = 1.5
W_TURN = 0.35
W_ARTERY_REBATE = 0.35
W_SIDE_EXTRA = 0.55
W_URGENCY_SIDE = 0.4
W_WAIT_BASE = 1.0
W_YIELD_WAIT = 0.2
W_NOWAIT_WAIT = 80.0
NOWAIT_MAX_WAIT = 0
YIELD_EXTRA_WAIT = 4
W_CONSTRUCTION_BLOCK = 1e6
W_CONSTRUCTION_SOFT = 25.0
W_ST_SOFT = 1.0

HIDDEN = 64
PARKING_THRESH = 0.5
SEMAPHORE_THRESH = 0.5
ACTIVE_PATH_HORIZON = 24
ARTERY_THRESH = 0.55


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, LOG_DIR, CKPT_DIR, CKPT):
        d.mkdir(parents=True, exist_ok=True)
