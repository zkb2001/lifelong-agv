from pathlib import Path

p = Path(__file__).with_name("solve_ecbs.py")
lines = p.read_text(encoding="utf-8").splitlines(True)
for ln in (4655, 4990, 5061, 5136, 5200, 5276, 5518):
    start = max(0, ln - 35)
    end = min(len(lines), ln + 25)
    out = Path(__file__).with_name(f"_snip_{ln}.txt")
    out.write_text("".join(f"{i+1:5d}|{lines[i]}" for i in range(start, end)), encoding="utf-8")
    print("wrote", out.name)
