from pathlib import Path
p = Path(r'd:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py')
lines = p.read_text(encoding='utf-8').splitlines()
# dump serial delivery need section and joint fail increment
for a,b in [(5160,5230),(4880,4960),(2400,2520)]:
    print(f'\n===== {a}-{b} =====')
    for i in range(a-1, min(b, len(lines))):
        print(f'{i+1}|{lines[i]}')
