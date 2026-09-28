import json
from pathlib import Path

root = Path(
    r"C:/Users/zkb17/.cursor/projects/d-compitition-MioVerse-final-version/agent-transcripts"
)
out = Path(
    r"d:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/_recover_ecbs"
)
# Search tool results that might contain file reads of solve_ecbs with solve_ecbs def
best = []
for p in root.rglob("*.jsonl"):
    for i, line in enumerate(p.open(encoding="utf-8", errors="replace"), 1):
        if "def solve_ecbs" not in line:
            continue
        if "use_hierarchical" not in line and "gate_tracker" not in line:
            continue
        # likely huge line
        if len(line) < 50000:
            continue
        best.append((len(line), str(p), i))

best.sort(reverse=True)
print("candidates", len(best))
for b in best[:10]:
    print(b)

# Also search for Read tool results with line numbers around solve_ecbs
# Extract from any line that has both use_lane_rules and def solve_ecbs
for p in root.rglob("*.jsonl"):
    for line in p.open(encoding="utf-8", errors="replace"):
        if "use_lane_rules" in line and "def solve_ecbs" in line and len(line) > 100000:
            try:
                obj = json.loads(line)
            except Exception:
                # raw save
                (out / "huge_line.txt").write_text(line[:2000], encoding="utf-8")
                print("raw huge", len(line), p)
                continue
            print("parsed huge", len(line), p)
            (out / "huge_parsed.json").write_text(json.dumps(obj)[:500], encoding="utf-8")
            break
