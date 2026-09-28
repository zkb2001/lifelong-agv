from pathlib import Path
p = Path(__file__).with_name("solve_ecbs.py")
lines = p.read_text(encoding="utf-8").splitlines(True)
# dump hold function + pickup commit area
for start, end, name in [(1080, 1160, "hold"), (4550, 4630, "pickup"), (4680, 4780, "deliv1")]:
    out = Path(__file__).with_name(f"_ctx_{name}.txt")
    out.write_text("".join(f"{i+1:5d}|{lines[i]}" for i in range(start-1, min(end, len(lines)))), encoding="utf-8")
    print(name)
