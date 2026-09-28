from pathlib import Path
p = Path(r'd:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py')
lines = p.read_text(encoding='utf-8').splitlines()
for i in range(650, 820):
    print(f'{i+1}|{lines[i]}')
