import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# Force only soft_ct
import ml_research.scripts.eval_ai_opt_v5 as E
E.MODES = [("soft_ct", True, False)]
slot = int(sys.argv[1])
sys.argv = ["eval_ai_opt_v5", str(slot)]
E.main()
