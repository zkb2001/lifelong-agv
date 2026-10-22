import os
import subprocess
import sys

seeds = ["0", "1", "2", "3", "4", "5", "7", "10", "42", "100", ""]
for seed in seeds:
    code = r"""
import os
os.environ['OMP_NUM_THREADS']='1'
from greed_coop_mcts_ref.scenario import load_slot_meta
from greed_coop_mcts_ref.solve import solve
meta=load_slot_meta(15, max_sim_time=200000, wall_timeout=1800.0, n_tasks=100)
meta=dict(meta); meta['id']='SH15_sweep'; meta['n_tasks']=100
rep=solve(15, meta=meta, wall_timeout=1800.0, max_sim_time=200000, policy='rrsc', device='cpu')
print('SEED', os.environ.get('PYTHONHASHSEED',''), rep.get('sim_time'), rep.get('n_stage_final'))
"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    if seed == "":
        env.pop("PYTHONHASHSEED", None)
    else:
        env["PYTHONHASHSEED"] = seed
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("SEED")]
    if lines:
        print(lines[-1])
    else:
        print("FAIL", seed, (r.stderr or "")[-400:])
