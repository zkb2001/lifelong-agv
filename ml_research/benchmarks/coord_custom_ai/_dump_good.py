from pathlib import Path

p = Path(__file__).with_name("solve_ecbs.py")
lines = p.read_text(encoding="utf-8").splitlines(True)
# already-good hold examples
for ln in (4625, 4768, 4932, 5249):
    start = max(0, ln - 40)
    end = min(len(lines), ln + 20)
    out = Path(__file__).with_name(f"_good_{ln}.txt")
    out.write_text("".join(f"{i+1:5d}|{lines[i]}" for i in range(start, end)), encoding="utf-8")
    print("wrote", out.name)
