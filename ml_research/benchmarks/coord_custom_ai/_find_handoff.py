from pathlib import Path
p = Path(r"d:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py")
text = p.read_text(encoding="utf-8")
print("size", p.stat().st_size, "lines", text.count("\n")+1)
for needle in ["AGV_M0_HANDOFF", "m0_handoff_enabled", "allow_yield=bool(allow_switch)", "allow_yield"]:
    print(needle, text.count(needle))
lines = text.splitlines()
for i,l in enumerate(lines):
    if "AGV_M0_HANDOFF" in l or "m0_handoff_enabled" in l or "allow_yield=bool(allow_switch)" in l:
        print(f"{i+1}:{l}")
# dump _solve_via_m0_engine start
for i,l in enumerate(lines):
    if "def _solve_via_m0_engine" in l:
        for j in range(i, min(i+80, len(lines))):
            print(f"{j+1}:{lines[j]}")
        break
print("--- call site ---")
for i,l in enumerate(lines):
    if "allow_yield=bool(allow_switch)" in l:
        for j in range(max(0,i-15), min(len(lines), i+10)):
            print(f"{j+1}:{lines[j]}")
