import json
from pathlib import Path

root = Path(__file__).resolve().parent
for p in sorted(root.glob("SH15*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    print(
        f"{p.name}: sim={d.get('sim_time')} done={d.get('tasks_completed')} "
        f"ok={d.get('validate_ok')} fleet={d.get('max_fleet_busy')} "
        f"cgpn={d.get('use_cgpn')} stage_f={d.get('n_stage_final')}"
    )
for sub in ["full", "no_ai", "no_stagenet", "no_rulenet", "no_pgfa", "no_pgcp"]:
    p = root / "ablation" / sub / "SH15.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"abl/{sub}: sim={d.get('sim_time')} done={d.get('tasks_completed')}")
p2 = Path(
    r"d:/compitition/MioVerse/final_version/greed_coop_mcts - 副本 (2)/results/SH15_gcm_gcm.json"
)
if p2.exists():
    d = json.loads(p2.read_text(encoding="utf-8"))
    print("copy2:", d.get("sim_time"), d.get("max_fleet_busy"), d.get("n_stage_final"))
