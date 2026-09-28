from pathlib import Path
p = Path(__file__).with_name("solve_ecbs.py")
lines = p.read_text(encoding="utf-8").splitlines(True)
for ln in (4105, 430, 350, 4675, 5262):
    start = max(0, ln - 40)
    end = min(len(lines), ln + 25)
    out = Path(__file__).with_name(f"_ctx_{ln}.txt")
    out.write_text("".join(f"{i+1:5d}|{lines[i]}" for i in range(start, end)), encoding="utf-8")
    print(out.name)
