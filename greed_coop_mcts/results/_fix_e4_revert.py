"""Strip E4 sticky remnants from pgcp.py; restore soft_leave path."""
from pathlib import Path
import re

p = Path(__file__).resolve().parents[1] / "pgcp.py"
t = p.read_text(encoding="utf-8")

# Remove E4 constants and helper
t = re.sub(
    r"\n# CEBI E4 \(SHARP-lite\):.*?\ndef e4_sticky_enabled\(.*?\n    \)\n",
    "\n",
    t,
    count=1,
    flags=re.S,
)
t = t.replace(
    "    egress_ttl: Dict[str, int] = field(default_factory=dict)  # E4 sticky ticks\n",
    "",
)
t = t.replace("    n_e4_sticky: int = 0\n", "")

# Restore simple pressure/just_here block
old = re.search(
    r"            pressure = bool\(carriers_by_goal\.get\(pad\)\).*?if just_here:\n                soft_leave\[n\] = pad\n                break",
    t,
    flags=re.S,
)
if not old:
    raise SystemExit("pressure block not found")
new = """            pressure = bool(carriers_by_goal.get(pad))
            just_here = last_unload_pad.get(n) == pad
            if pressure:
                egress[n] = pad
                break
            if just_here:
                soft_leave[n] = pad
                break"""
t = t[: old.start()] + new + t[old.end() :]

# Restore fleet_cebi without n_ag/near if we added them — keep n_ag/near ok
t = t.replace(
    "    n_ag = len(positions)\n    near = int(near_pad_deg2)\n"
    "    # Extreme-throat CEBI gate (map-id free).\n"
    "    fleet_cebi = n_ag >= 10 and near >= 100\n",
    "    # Extreme-throat CEBI gate (map-id free).\n"
    "    fleet_cebi = len(positions) >= 10 and int(near_pad_deg2) >= 100\n",
)

p.write_text(t, encoding="utf-8")
print("reverted E4 from pgcp.py")
