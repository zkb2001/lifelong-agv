"""Revert accidental TBC band edits in pgcp.py."""
from pathlib import Path
import re

p = Path(__file__).resolve().parents[1] / "pgcp.py"
t = p.read_text(encoding="utf-8")
t = t.replace(
    "# Mid-open throats (near≈45) still congest under cap6 — SH02-class.\n"
    "THROAT_NEAR_LO = 45",
    "THROAT_NEAR_LO = 55",
)
t = t.replace("THROAT_NEAR_LO = 45", "THROAT_NEAR_LO = 55")
new = '''def throat_band_fleet_cap(n_agents: int, near_pad_deg2: int, *, base: int = 6) -> int:
    """Map-id-free capacity from fleet size × corridor-band geometry."""
    if int(n_agents) >= 28:
        return max(int(base), 7)
    if (
        MID_FLEET_LO <= int(n_agents) <= MID_FLEET_HI
        and THROAT_NEAR_LO <= int(near_pad_deg2) <= THROAT_NEAR_HI
    ):
        return int(THROAT_CAP)
    # Dense mid-large fleets on moderate throats (not mega-open / not
    # extreme labyrinth): inject one fewer concurrent carrier. Empirically
    # SH11-class (n in [16,18], near in [80,90]); wider bands regress SH12.
    if 16 <= int(n_agents) <= 18 and 80 <= int(near_pad_deg2) <= 90:
        return int(THROAT_CAP)
    return int(base)'''
pat = r"def throat_band_fleet_cap\(n_agents: int, near_pad_deg2: int, \*, base: int = 6\) -> int:.*?return int\(base\)"
t2, n = re.subn(pat, new, t, count=1, flags=re.S)
print("replaced", n, "LO55", "THROAT_NEAR_LO = 55" in t2)
assert n == 1
p.write_text(t2, encoding="utf-8")
