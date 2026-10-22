import json
from pathlib import Path

root = Path(__file__).resolve().parent
for name in [
    "SH12_gcm_gcm.json",
    "SH12_cgpn_gcm.json",
    "SH08_cgpn_gcm.json",
    "SH11_cgpn_gcm.json",
    "SH03_cgpn_gcm.json",
    "SH15_abl_full_gcm.json",
]:
    p = root / name
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(
            name,
            "sim",
            d.get("sim_time"),
            "done",
            d.get("tasks_completed"),
            "ok",
            d.get("validate_ok"),
            "stage_f",
            d.get("n_stage_final"),
        )

full = {
    1: 965,
    2: 2021,
    3: 4440,
    4: 1053,
    5: 1616,
    6: 2024,
    7: 3897,
    8: 1320,
    9: 1337,
    10: 2215,
    11: 2384,
    12: 1558,
    13: 832,
    14: 1216,
    15: 2523,
    16: 1643,
    17: 2395,
    18: 1711,
    19: 1446,
    20: 947,
}
dirs = [
    p
    for p in Path(r"d:/compitition/MioVerse/final_version").iterdir()
    if p.is_dir() and p.name.startswith("greed") and p.name != "greed_coop_mcts"
]
b = dirs[0] / "results"
sims = []
print("=== copy2 ===")
for i in range(1, 21):
    p = b / f"SH{i:02d}_gcm_gcm.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        sim = int(d.get("sim_time"))
        print(f"SH{i:02d} {sim} dlt={sim - full[i]:+d}")
        sims.append(sim)
print("copy2 avg", sum(sims) / len(sims) if sims else None)
