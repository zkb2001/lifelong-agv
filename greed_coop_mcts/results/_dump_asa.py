"""Dump ASA / TBC fields from latest summary.json."""
import json
from pathlib import Path

rows = json.loads(
    (Path(__file__).resolve().parent / "summary.json").read_text(encoding="utf-8")
)["rows"]
for r in rows:
    print(
        f"SH{int(r['slot']):02d} sim={r.get('sim_time')} "
        f"asa={r.get('asa_stage_net')} near={r.get('near_pad_deg2')} "
        f"fleet={r.get('max_fleet_busy')} ok={r.get('validate_ok')}"
    )
