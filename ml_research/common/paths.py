"""Project paths for ML research experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = Path(__file__).resolve().parents[1]
RESULTS = ML_ROOT / "results"
DATA = ML_ROOT / "data"
CKPT = ML_ROOT / "checkpoints"

# Shared competition / simulation data (project root)
DATA_DIR = ROOT / "data"
TRAJ_DIR = DATA_DIR / "outputs" / "trajectories"

POSITION_CSV = DATA_DIR / "agv_position.csv"
TASK_CSV = DATA_DIR / "agv_task.csv"
TASK_EXTREME_CSV = DATA_DIR / "agv_task_1.csv"
MAIN_COPY = ROOT / "simulation" / "engine.py"

TRAJECTORIES = [
    TRAJ_DIR / "agv_trajectory.csv",
    TRAJ_DIR / "agv_trajectory_normal.csv",
    TRAJ_DIR / "agv_trajectory_extreme.csv",
    TRAJ_DIR / "agv_trajectory_prob.csv",
]

GRID_SIZE = 21
LOCAL_R = 3  # local observation radius -> (2R+1)^2 = 7x7

for d in (RESULTS, DATA, CKPT):
    d.mkdir(parents=True, exist_ok=True)
