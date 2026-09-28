from pathlib import Path
p = Path(r'd:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py')
lines = p.read_text(encoding='utf-8').splitlines()
out = []
for i in range(3408, 3520):
    out.append(f'{i+1}|{lines[i]}')
Path(r'd:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/_probe.txt').write_text('\n'.join(out), encoding='utf-8')
print('ok')
