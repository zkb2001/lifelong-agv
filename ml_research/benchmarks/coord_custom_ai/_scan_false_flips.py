from pathlib import Path
lines = Path(r"d:\compitition\MioVerse\final_version\ml_research\benchmarks\coord_custom_ai\solve_ecbs.py").read_text(encoding="utf-8").splitlines()
needle = '[-1]["loaded"] = "FALSE"'
for i, l in enumerate(lines):
    if needle in l:
        print(f"==== line {i+1}")
        for j in range(max(0, i - 10), min(len(lines), i + 15)):
            print(f"{j+1}:{lines[j]}")
