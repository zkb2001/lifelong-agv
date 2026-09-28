from pathlib import Path
p = Path(r'd:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py')
lines = p.read_text(encoding='utf-8').splitlines()
for i,l in enumerate(lines):
    if 'ST parallel' in l or 'Leftover' in l or 'leftover' in l or 'joint fail/skip' in l:
        print(f'{i+1}|{l}')
        for j in range(i, min(i+15, len(lines))):
            if j != i:
                print(f'{j+1}|{lines[j]}')
        print('---')
